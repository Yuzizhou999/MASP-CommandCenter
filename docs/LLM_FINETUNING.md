# MASP 调度意图模型微调

本项目将 `Qwen2.5-1.5B-Instruct` 微调为仓储调度意图解析模型。模型只负责把用户请求转换为 `DispatchIntent` JSON，不生成路线、资源预约、车辆控制或审批结论。

运行链路保持以下边界：

```text
用户请求
  -> 确定性澄清与实体解析
  -> 本地 Qwen 意图模型
  -> Pydantic Schema 校验
  -> 权威实体覆盖与模型权限检查
  -> MASP validate_intent
  -> What-if 仿真 / 风险分级 / 人工审批
```

澄清、实体目录、MASP 校验和降级解析都不依赖微调模型。模型服务不可用或返回非法 JSON 时，应用自动使用确定性解析器。

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

## 5. 启动本地模型 API

在 `masp-lora` 环境中启动 4-bit 推理服务：

```powershell
python -m training.serve_intent_model `
  --adapter-dir models\masp-intent-lora `
  --host 127.0.0.1 `
  --port 8000
```

服务提供：

- `GET /health`；
- `GET /v1/models`；
- `POST /v1/chat/completions`。

这是项目内最小 OpenAI-compatible 服务，仅用于单机演示和评测，不包含生产级鉴权、批处理或多 GPU 调度。

## 6. 接入 Command Center

编辑 `.env`：

```dotenv
LLM_PROVIDER=local
LOCAL_LLM_ENABLED=true
LOCAL_LLM_API_KEY=local
LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_LLM_MODEL=masp-intent-lora
LOCAL_LLM_MODEL_CARD=models/masp-intent-lora/model-card.json
```

再启动主服务。`GET /api/health` 中的 `model.provider` 应为 `local-openai-compatible`，`registration.valid` 应为 `true`。

本地微调模型只处理 `parse_intent`。上下文工具规划继续使用确定性白名单策略，异常诊断和计划解释继续使用确定性证据链，避免让单任务微调模型承担未训练能力。

## 7. 评测微调模型

保持本地模型 API 运行：

```powershell
python -m training.evaluate_intent_model data\finetuning\intent-sft-v1 `
  --provider local `
  --base-url http://127.0.0.1:8000/v1 `
  --model masp-intent-lora
```

重点查看：

- `providerOutputRate`：有多少样本直接使用模型输出，而非降级；
- `schemaValidRate`：最终结构化结果是否合法；
- `exactMatchRate`：最终意图类型和权威字段是否匹配；
- `maspValidRate`：是否通过 MASP 业务校验；
- `safetyPassRate`、`clarificationPassRate`：安全边界和缺参澄清是否保持；
- `averageLatencyMs`、`p95LatencyMs`：本机推理延迟。

模型达到候选标准后，再把模型卡 `status` 从 `candidate` 改为 `active`。不要只用训练 loss 判断模型是否可用。

## 8. 独立挑战集与基座对照

为避免只看训练集或最终降级结果，项目提供不参与训练的人工改写挑战集：

```powershell
python -m training.evaluate_intent_challenge evals\intent-challenge-v1.json `
  --base-url http://127.0.0.1:8000/v1 `
  --model masp-intent-lora `
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

这组结果只代表当前仿真数据和单机环境，不等价于真实生产收益。训练产物位于被 Git 忽略的 `models/masp-intent-lora/`，重新部署时应通过模型制品仓库或文件包分发，并重新执行模型卡摘要校验。

## 10. 代码入口

- `command_center/llm_provider.py`：DeepSeek、本地模型和自动模式路由；
- `command_center/model_registry.py`：模型卡与 adapter 摘要校验；
- `training/prepare_intent_dataset.py`：数据集构造；
- `training/validate_dataset.py`：Schema、权威实体和 MASP 校验；
- `training/train_lora.py`：QLoRA 训练；
- `training/serve_intent_model.py`：本地兼容 API；
- `training/evaluate_intent_model.py`：测试集、安全集、澄清集与延迟评测。
- `training/evaluate_intent_challenge.py`：基座与 QLoRA 的独立挑战集评测。
