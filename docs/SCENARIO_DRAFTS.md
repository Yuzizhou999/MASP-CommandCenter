# 场景草稿生命周期

CommandCenter第二阶段提供了场景包的编辑、校验、任务生成、编译和发布接口。场景包是地图、车辆、工作站、交通资源和任务流的统一JSON契约，接口不会直接修改MASP仓库内的固定运行场景。

## 接口

- `POST /api/v1/scenario-drafts` 创建草稿；
- `GET /api/v1/scenario-drafts` 和 `GET /api/v1/scenario-drafts/{packageId}` 查询草稿；
- `PUT /api/v1/scenario-drafts/{packageId}?expectedRevision=N` 更新草稿；
- `POST /api/v1/scenario-drafts/{packageId}/validate` 执行Schema和领域校验；
- `POST /api/v1/scenario-drafts/{packageId}/generate-tasks?expectedRevision=N` 按固定间隔、泊松或分时波峰生成任务；
- `POST /api/v1/scenario-drafts/{packageId}/compile` 编译为隔离的MASP仿真资产；
- `POST /api/v1/scenario-drafts/{packageId}/publish` 发布不可变仿真版本。

写操作会记录到CommandCenter审计日志。更新和任务生成必须携带当前`revision`，版本不一致返回409，避免画布或多人编辑互相覆盖。发布前必须通过场景包Schema、确定性领域校验并成功编译；发布版本写入`data/scenario-builds/{packageId}/{version}/published`，当前仅供数字孪生仿真使用，不连接真实车辆控制器。
