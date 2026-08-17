# ScenarioPackage 第一阶段

`ScenarioPackage` 是仓储地图、车队、交通资源、任务流和异常事件的统一输入契约。它位于可视化编辑器和 MASP 调度内核之间：编辑器只修改场景包，编译器负责生成 MASP 当前已经使用的地图、冲突资源、工位、车型、车辆、交通区和调度场景文件。

## 当前交付

- `command_center/masp/scenario_package.py`：契约对象、确定性校验、固定资产迁移和运行文件编译；
- `schemas/scenario-package.schema.json`：JSON Schema，供前端和后端在保存草稿时预校验；
- `tests/test_scenario_package.py`：覆盖迁移、Schema、非法引用、重复起点和编译结果。

## 契约结构

`WarehouseSceneSpec` 包含仓库边界和节点坐标、按车型区分的节点位置和等待权限、有向贝塞尔道路、工位能力、车辆初始位置、窄路区、恢复点和安全参数。

`TaskStreamSpec` 包含固定随机种子、仿真结束时间、任务释放时间、起终点、车型、载荷、优先级、时限，以及紧急插单、封路、工位降容和故障等时间事件。

## 校验边界

发布前会检查节点和道路ID唯一性、道路端点和车型能力、曲线几何、工位覆盖、车辆起点等待权限、任务时序、车型匹配、定向可达性、放货后的恢复点可达性、交通区引用和事件时间。冲突资源由车辆外形扫掠几何重新生成，不接受编辑器直接提交的冲突表。

## 从现有场景迁移

在 `E:\project\MASP-CommandCenter` 启动后，通过场景设计工作台选择一个MASP运行场景，点击“从运行场景创建”；也可以调用：

```powershell
Invoke-RestMethod -Method Post `
  "http://127.0.0.1:8877/api/v1/scenario-drafts/from-runtime?scenarioId=realistic-multi-fleet&packageId=realistic-draft"
```

命令会读取当前固定地图和 `realistic-multi-fleet.json`，写出一个可编辑的草稿包和以下编译产物：

```text
map-model.json
conflict-resources.json
workstations.json
robot-profiles.json
scheduler.json
initial-vehicles.json
traffic-zones.json
dispatch-scenario.json
task-stream.json
validation-report.json
manifest.json
```

使用已有场景包重新编译：

```powershell
Invoke-RestMethod -Method Post `
  "http://127.0.0.1:8877/api/v1/scenario-drafts/realistic-draft/compile"
```

编译产物可以直接传给现有 `tools/simulate_dispatch.py`，不需要修改 SIPP、RH-PP 或资源预约实现。

## 设计边界

场景契约、参数化任务流生成器和画布均由比赛仓库维护。原MASP仓库只提供既有调度与数字孪生运行时，不包含比赛项目新增代码。大模型只负责把自然语言转换为场景草稿参数，不能直接生成或写入路线、预约表和安全控制指令。
