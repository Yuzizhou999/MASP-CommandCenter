# MASP 调度 Agent 模型微调

项目保留两套可对照能力：冻结的 v1 adapter 把用户请求转换为 `DispatchIntent` JSON；v2 候选 adapter 学习有界 Agent 的单动作协议，在每轮从 `CALL_TOOL`、`REQUEST_CLARIFICATION`、`PROPOSE_INTENT` 中选择一个动作。两者都不生成路线、资源预约、车辆控制或审批结论。

运行链路保持以下边界：

```text
用户请求
  -> 确定性澄清与权威实体绑定
  -> 本地 Qwen / DeepSeek / deterministic driver
  -> CALL_TOOL -> observation -> 下一轮决策
  -> PROPOSE_INTENT -> Pydantic 与权威边界
  -> MASP validate_intent -> 可修复 issue 最多回送两次
  -> What-if 仿真 / 风险分级 / 人工审批
```

硬缺参澄清、实体目录、工具白名单、检索注入隔离、MASP 校验、预算和审批都不依赖微调模型。非法工具或非法 JSON 会成为 rejection observation；本地服务不可用时应用使用确定性 driver。

`models/masp-intent-lora` 是冻结 v1 基线，不能被 v2 训练覆盖。v2 只有同时通过原意图挑战集无退化门槛和新增轨迹门槛后才能晋级。

## 1. 独立环境

不要把训练依赖装入项目运行环境或现有 `dfjsp2t` 环境。建议创建独立环境：

```powershell
conda create -n masp-lora python=3.10 -y
conda activate masp-lora
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-finetune.txt
```

确认 CUDA 和 BF16：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
```

当前 8GB 显存配置使用 4-bit NF4、batch size 1、梯度累积 16 和 gradient checkpointing。若 Windows 下 `bitsandbytes` 无法加载 CUDA，使用 WSL2 Ubuntu 创建同名独立环境，代码目录可从 `/mnt/e/project/MASP-CommandCenter` 访问。

## 2. 生成训练数据

数据只从锁定的 `E:\project\MASP-locked` 场景和人工维护的领域模板生成，不修改原始 MASP 仓库：

```powershell
$env:MASP_ENGINE_ROOT='E:\project\MASP-locked'
python -m training.prepare_intent_dataset --output-dir data\finetuning\intent-sft-v1
python -m training.validate_dataset data\finetuning\intent-sft-v1
```

生成内容包括：

- `intent-sft-train.jsonl`、`intent-sft-valid.jsonl`、`intent-sft-test.jsonl`；
- 独立的安全攻击 holdout 和缺参澄清 holdout；
- 固定随机种子、引擎提交、场景列表、数量和文件 SHA-256 的 `manifest.json`。

同一个任务实体或封闭资源的不同表达使用同一 split key，避免同实体模板泄漏到训练集和测试集。训练样本与生产推理复用 `intent_training_messages()`，降低训练和运行 Prompt 不一致的问题。

## 3. 运行确定性基线

训练前先保存不调用模型的基线：

```powershell
python -m training.evaluate_intent_model data\finetuning\intent-sft-v1 --provider deterministic
```

该基线用于确认数据、Schema、MASP 校验、澄清和安全降级链路本身是正确的。它不是微调效果指标。

## 4. QLoRA 训练

```powershell
python -m training.train_lora data\finetuning\intent-sft-v1 `
  --config training\configs\intent-lora.json `
  --output-dir models\masp-intent-lora
```

训练只对 assistant completion 计算 loss。配置使用：

- 基座：`Qwen/Qwen2.5-1.5B-Instruct`；
- 量化：4-bit NF4 + double quantization；
- LoRA：`r=16`、`alpha=32`、dropout `0.05`；
- 上下文：2048 tokens，足以容纳运行时 Schema 和回答；
- 训练：2 epochs，学习率 `2e-4`，有效 batch size 16。

产物目录包含 adapter、tokenizer、训练配置和 `model-card.json`。模型卡登记基座、数据集版本、训练指标和 adapter SHA-256；应用健康接口会验证该摘要。

### 多轮 Agent v2

v2 数据不是由 1.5B 自滚动生成，而是由冻结的确定性策略、v1 保持样本、人工设计的工具拒绝/修复/澄清/注入轨迹，以及可选的人工审核 teacher 轨迹组成。独立 gold 不进入训练集。

```powershell
$env:MASP_ENGINE_ROOT='E:\project\MASP-locked'
python -m training.prepare_agent_dataset `
  --source-dir data\finetuning\intent-sft-v1 `
  --output-dir data\finetuning\agent-sft-v2
python -m training.validate_dataset data\finetuning\agent-sft-v2
python -m training.train_lora data\finetuning\agent-sft-v2 `
  --config training\configs\agent-lora-v2.json `
  --output-dir models\masp-agent-lora-v2
```

训练器会监督多轮会话中选定的 assistant turn；故意构造的非法动作可通过 `superviseAssistantIndices` 排除。`manifest.json` 固定数据摘要、MASP commit、生成策略和 gold 隔离声明。

训练前必须对全部 split 做 token 预检，不能通过尾部截断静默丢弃 system prompt、用户请求或权威参数：

```powershell
python -m training.tokenization_preflight data\finetuning\agent-sft-v2 `
  --config training\configs\agent-lora-v2.json `
  --output models\masp-agent-lora-v2\tokenization-preflight.json
```

只要任一样本超过 `maxLength`，预检和训练都会直接失败并报告 `exampleId`、类别和实际 token 数。当前 v2/v2.1 数据最大长度为 1423，配置固定为 2048，不发生截断。训练环境要求 `transformers 4.x`；预检会拒绝 5.x，避免聊天模板返回类型变化造成错误长度统计。

默认仍按 epoch 保存 checkpoint 并加载最佳模型。只有运行环境存在明确的会话时限时，才传入 `--checkpoint-steps N` 改为定步保存；恢复时使用 `--resume-from-checkpoint`。这个选项不改变未传参时的历史训练行为。

### v2.2 失败驱动受控实验

v2.2 不直接与历史单次训练数字归因比较。先从 `agent-lora-v2.json` 复制控制配置，用原 v2 数据训练 `v2-repro`；再用相同学习率 `1.5e-4`、梯度累积 16、seed `20260825` 和 2048 token 训练 v2.2，唯一数据变量是 v2.1 新增的 34 条失败驱动样本：

```powershell
python -m training.prepare_agent_dataset_v21
python -m training.tokenization_preflight data\finetuning\agent-sft-v2.1 `
  --config training\configs\agent-lora-v2.2.json
python -m training.train_lora data\finetuning\agent-sft-v2.1 `
  --config training\configs\agent-lora-v2.2.json `
  --output-dir models\masp-agent-lora-v2.2
```

本机完整训练记录为 `train_loss=0.015333`、`eval_loss=0.003809`、最大显存 4879.9 MB，`maxObservedTokens=1423`、`truncatedExamples=0`。冻结评测观察到 1/6 个目标案例改善且无 case 回归，但意图无退化门、工具 recall、修复门和边界拦截门未通过，因此 `v2.2` 保持 `candidate`，默认仍是 v1。

### v2.3 单协议稳定化实验

v2.3 用统一的 `AgentAction` envelope 训练每一个受监督目标，不再混入裸 `DispatchIntent` completion。数据同时覆盖工具选择、服务端生成澄清、意图提出、非法协议恢复、Schema 恢复和 verifier 修复状态：

```powershell
python -m training.prepare_agent_dataset_v23
python -m training.tokenization_preflight data\finetuning\agent-sft-v2.3 `
  --config training\configs\agent-lora-v2.3.json
python -m training.train_lora data\finetuning\agent-sft-v2.3 `
  --config training\configs\agent-lora-v2.3.json `
  --output-dir models\masp-agent-lora-v2.3
```

本次数据共 1716 条，train/valid/test 为 1294/203/219，裸意图目标为 0；`maxObservedTokens=1403`，在 `maxLength=2048` 下无截断。单 seed `20260827` 完整训练得到 `train_loss=0.023536`、`eval_loss=0.002458`，峰值显存 4844.2 MB。

推理约束的审计边界必须保留：训练前方案写的是 LM Format Enforcer，但训练后发现 LMFE 无法稳定编译动作联合 Schema，并可能在合法动作完成前提前 EOS。因此正式对照改为让控制组和候选组都使用同一 XGrammar 七分支 Schema，生成后再由 `jsonschema` 终检。这个改动保证两组同口径，但它是训练后的推理修订，所以 v2.3 是稳定化实验，不是预注册的单变量消融。

原始意图挑战中，候选相对控制的 Macro F1 为 `0.7457 -> 0.8294`、精确匹配为 `0.72 -> 0.78`、槽位匹配为 `0.60 -> 0.90`、MASP 有效率为 `0.78 -> 0.86`。当时 18 条轨迹 holdout 的目标完成率都为 `0.7222`，工具 recall 从 `0.9167` 降至 `0.8611`，model-driven rate 从 `1.0` 降至 `0.9231`。资格结论是 `KEEP_V1`，因此没有继续训练另外两个 seed，v2.3 模型卡保持 `candidate`。

后续审计确认原轨迹结果混入 resolver 词表、中文时长、`ungrounded` 终态和槽位评分缺陷。先运行不调用模型的系统可达性预检；低于 suite 的目标成功率门槛时命令以退出码 2 失败，禁止继续消耗 GPU：

```powershell
python -m training.preflight_agent_system `
  --suite evals\agent-trajectories-v2.1-holdout.json `
  --output results\agent-system-preflight-v21.json
```

修复后预检为 18/18、理论上限 `1.0`。两个既有 adapter 未重训，只在相同 AgentAction prompt、XGrammar Schema 和请求集合下重评。control 轨迹目标成功率为 `0.8333`，v2.3 为 `0.8889`，但候选工具 recall `0.8333`、澄清准确率 `0.9444`、边界拦截 recall `0.50`，仍然 `KEEP_V1`。最新 intent 重评为 Macro F1 `0.7457 -> 0.7991`、精确匹配 `0.72 -> 0.78`、槽位匹配 `0.60 -> 0.80`；历史 `0.8294` 没有稳定复现。训练已经停止，后续只在系统可达性、评测契约哈希和冻结门槛全部通过后才考虑新候选。

## 5. 启动本地模型 API

在 `masp-lora` 环境中启动 4-bit 推理服务：

```powershell
python -m training.serve_intent_model `
  --adapter-dir models\masp-agent-lora-v2.3 `
  --host 127.0.0.1 `
  --port 8000 `
  --require-xgrammar
```

正式运行受约束 Agent 评测时增加 `--require-xgrammar`。该参数会在 XGrammar 不可用时直接拒绝启动，防止把只有生成后 Schema 终检的兼容模式误当成 token 级约束评测。

服务提供：

- `GET /health`；
- `GET /v1/models`；
- `POST /v1/chat/completions`。

本地 AgentAction 请求可以携带严格 JSON Schema。安装 XGrammar 时，服务约束 token 生成，并在返回前使用 `jsonschema` 终检；无法满足 Schema 时返回 422，不把非法输出伪装成成功结果。没有 XGrammar 的演示环境只做生成后终检，`/health` 会明确报告 `jsonschema-validation-only`；正式评测必须用 `--require-xgrammar`。模型仍在 GPU 上推理；没有 C 编译器的 WSL 环境中，grammar bitmask 在 CPU 计算。

这是项目内最小 OpenAI-compatible 服务，仅用于单机演示和评测，不包含生产级鉴权、批处理或多 GPU 调度。

## 6. 接入 Command Center

编辑 `.env`：

```dotenv
LLM_PROVIDER=local
LOCAL_LLM_ENABLED=true
LOCAL_LLM_API_KEY=local
LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_LLM_MODEL=masp-agent-lora-v2.3
LOCAL_LLM_MODEL_CARD=models/masp-agent-lora-v2.3/model-card.json
AGENT_RUNTIME_MODE=loop
```

再启动主服务。`GET /api/health` 中的 `model.provider` 应为 `local-openai-compatible`，`registration.valid` 应为 `true`。

当前演示不再并排运行 v1 与 v2，也不再维护两个后端端口。`scripts/start.ps1` 固定启动一个 v2.3/XGrammar 模型服务和一个 loop 后端。代码中的 linear driver 只用于历史报告复算和回归测试，不是当前演示部署路径。

## 7. 评测微调模型

保持本地模型 API 运行：

```powershell
python -m training.evaluate_intent_model data\finetuning\intent-sft-v1 `
  --provider local `
  --base-url http://127.0.0.1:8000/v1 `
  --model masp-agent-lora-v2.3
```

重点查看：

- `providerOutputRate`：有多少样本直接使用模型输出，而非降级；
- `schemaValidRate`：最终结构化结果是否合法；
- `exactMatchRate`：最终意图类型和权威字段是否匹配；
- `maspValidRate`：是否通过 MASP 业务校验；
- `safetyPassRate`、`clarificationPassRate`：安全边界和缺参澄清是否保持；
- `averageLatencyMs`、`p95LatencyMs`：本机推理延迟。

不要因为它成为唯一演示模型就把模型卡门禁结果改写为通过；训练 loss 和单模型打包方式都不能替代轨迹评测。

### 冻结轨迹评测与晋级

轨迹 gold 独立标注了必需/允许/禁止工具、终态、是否应澄清、意图类型、可修复 verifier issue 和直接/间接注入。评测器会在指定 case 的第一次 MASP 校验中注入 gold issue，确保 `repairSuccessRate` 测到真实回路，而不是空指标。

当前服务只直接回放 v2.3。历史 v1/v2-repro 对照报告仍可交给 `training.compare_agent_replays` 和 `training.qualify_agent_candidate` 离线复算，但不需要同时启动第二个模型服务。

当前正式轨迹回归使用 100 条分层 suite。先执行确定性可达性预检，再分别用同一个 XGrammar 服务契约回放 control 与 candidate：

```powershell
$env:MASP_ENGINE_ROOT='E:\project\MASP-locked'

python -m training.preflight_agent_system `
  --suite evals\agent-trajectories-v3-stratified.json `
  --output results\agent-system-preflight-v3-stratified.json

python -m training.evaluate_agent_trajectories `
  --suite evals\agent-trajectories-v3-stratified.json `
  --mode loop_local `
  --local-base-url http://127.0.0.1:8000/v1 `
  --local-candidate-model masp-agent-lora-v2.3 `
  --output-dir results\agent-eval-v3-stratified-candidate
```

回放前必须确认 `/health` 返回 `masp-agent-lora-v2.3` 和 `structuredOutput=xgrammar`。与历史对照报告比较时，仍必须核对 `suiteSha256`、`promptSha256ByMode.loop_local`、`evaluationContractSha256` 和 `requestPromptSetSha256`。100 条 suite 将单条波动降为 1%，但它是训练后编写的分层回归，同层样本也不是完全独立；它适合做回归门禁，不应包装成盲测统计证明。

## 8. 独立挑战集与基座对照

为避免只看训练集或最终降级结果，项目提供不参与训练的人工改写挑战集：

```powershell
python -m training.evaluate_intent_challenge evals\intent-challenge-v1.json `
  --base-url http://127.0.0.1:8000/v1 `
  --model masp-agent-lora-v2.3 `
  --output data\finetuning\challenge-lora.json
```

基座对照使用同一个服务入口，但不加载 adapter：

```powershell
python -m training.serve_intent_model `
  --base-model Qwen/Qwen2.5-1.5B-Instruct `
  --model-id qwen2.5-1.5b-base `
  --host 127.0.0.1 `
  --port 8000
```

挑战集包含 50 条均衡的正常请求（5 类意图各 10 条）、10 条越权/危险请求和 10 条缺参澄清请求。评测器记录模型未经修正的原始输出，不会替模型补写任务或资源槽位。结果分为两组：

- `qualification.model`：请求成功率、JSON/schema 有效率、意图宏 F1、槽位匹配、MASP 校验和延迟，衡量微调模型的解析能力；
- `qualification.system`：模型调用前的确定性安全门召回率和澄清准确率，衡量部署边界；
- `diagnostics.rawSafetyPassRate`：模型自己拒绝危险请求的比例，仅作诊断，不能作为上线安全依据。

在本机 RTX 5060 Laptop 8GB 显存上，同一挑战集的实测对照如下：

| 指标 | Qwen 基座 | QLoRA adapter |
| --- | ---: | ---: |
| JSON 有效率 | 2% | 100% |
| Schema 有效率 | 2% | 100% |
| 意图准确率 | 0% | 94% |
| 意图宏 F1 | 0 | 0.937 |
| 任务/资源槽位精确率 | 0% | 100% |
| MASP 校验通过率 | 2% | 98% |
| P95 延迟 | 8555ms | 6592ms |
| 模型原始安全率 | 0% | 0% |
| 确定性安全门召回率 | 100% | 100% |
| 澄清准确率 | 100% | 100% |

QLoRA 通过模型和系统两组资格门；基座模型只通过系统组。QLoRA 的 50 条正常样本中有 3 条报告请求被分类为其他意图（`RPT-002`、`RPT-003`、`RPT-009`），这是当前已知的模型误差。两份完整 JSON 报告默认写入被 Git 忽略的 `data/finetuning/challenge-*.json`，便于本机复核。

安全结论必须这样表述：模型本身不是可信执行者，危险请求即使被模型解析成 `CREATE_TASK` 或其他结构，也会在模型调用前被确定性安全门拦截，之后仍需 Schema、权威实体覆盖、MASP 校验、What-if 仿真和审批。模型原始安全率为 0% 是有意保留的风险证据，不应隐藏或改写成 100%。

## 9. 当前已完成实例

本机 RTX 5060 Laptop（8GB 显存）已完成一次真实训练：

- 数据集：766 条，train/valid/test 为 566/98/102，锁定 MASP 提交 `ab431c9ee3283071d1d13be0a174f2259b671687`；
- 训练：约 36 分钟，4-bit NF4 QLoRA，2 epochs；Trainer 自动选择第 1 epoch 的最佳 checkpoint；
- 训练指标：`train_loss=0.006366`，`eval_loss=0.005049`；
- 端到端测试：模型输出率、Schema 有效率、精确字段匹配率、MASP 有效率、安全 holdout、澄清 holdout 均为 `1.0`；这里的安全/澄清指标是经过确定性边界和降级链路后的系统结果，不代表模型原始拒答能力；
- 本机延迟：平均约 `4855ms`，P95 约 `8582ms`；
- adapter：`73.9MB`，模型卡状态为 `active`，SHA-256 已由应用健康接口复核。

这组结果只代表当时的仿真数据和单机环境，不等价于真实生产收益。v1 本地 adapter 已从当前演示工件中删除，历史评测报告仍保留；当前重新部署只分发 `models/masp-agent-lora-v2.3/`，并重新执行模型卡摘要校验。

## 10. 代码入口

- `command_center/llm_provider.py`：DeepSeek、本地模型和自动模式路由；
- `command_center/model_registry.py`：模型卡与 adapter 摘要校验；
- `training/prepare_intent_dataset.py`：数据集构造；
- `training/validate_dataset.py`：Schema、权威实体和 MASP 校验；
- `training/train_lora.py`：QLoRA 训练；
- `training/serve_intent_model.py`：本地兼容 API；
- `training/evaluate_intent_model.py`：测试集、安全集、澄清集与延迟评测。
- `training/evaluate_intent_challenge.py`：基座与 QLoRA 的独立挑战集评测。
- `training/prepare_agent_dataset.py`：多轮单动作轨迹数据构造；
- `training/prepare_agent_dataset_v23.py`：统一 AgentAction 目标与恢复状态数据构造；
- `training/prepare_agent_eval_v3.py`：生成 100 条、10 分层的冻结轨迹回归集；
- `training/preflight_agent_system.py`：训练或评测前计算确定性系统可达上限与 suite 质量；
- `training/evaluate_agent_trajectories.py`：轨迹、修复和注入评测；
- `training/compare_agent_replays.py`：同 case paired replay diff；
- `training/compare_agent_experiment.py`：v2-repro 与 v2.2 的受控同 case 对比；
- `training/qualify_agent_candidate.py`：v1 无退化与 v2 新增量的双层晋级门。
