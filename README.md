# BVH Processing API

异步接收 BVH 处理任务，从 MinIO 下载原始 BVH，并通过 `multipart/form-data` 将处理结果回调给业务后端。当前联调版本暂不修改文件内容，后续平滑算法统一接入 `services/processing.py`。

## 环境要求

- Python 3.12+
- uv

## 安装与启动

```bash
uv sync
uv run uvicorn bvh_processing.main:app \
  --host 0.0.0.0 \
  --port 9001
```

启动后可访问：

- Swagger 文档：`http://127.0.0.1:9001/docs`
- 健康检查：`http://127.0.0.1:9001/health`

## 提交处理任务

### `POST /api/v1/bvh/process`

请求：

```json
{
  "actionId": "action-42",
  "originalFileUrl": "https://minio.example.com/bucket/walk.bvh",
  "handleOptions": [1, 2, 3],
  "callbackUrl": "https://backend.example.com/callbacks/bvh"
}
```

字段说明：

- `actionId`：业务后端的动作记录 ID，回调时原样返回。
- `originalFileUrl`：可直接下载的 MinIO BVH 地址。
- `handleOptions`：按顺序执行的整数 JSON 数组。编号含义：`1` 整体去噪、
  `2` 整体平滑、`3` 脚步锁定校正、`4` 循环优化。
- `callbackUrl`：处理完成后的回调地址。
- 请求中不接收 `callbackToken`。

任务接收成功后立即返回：

```json
{
  "success": true,
  "taskId": "a37d60af-6b3d-4612-b88e-3ba1e4e24f6f",
  "message": "任务已接收"
}
```

这里的 `success` 只表示任务是否接收成功，不代表最终处理结果。

参数校验失败时返回 HTTP 422：

```json
{
  "success": false,
  "taskId": null,
  "message": "请求参数不正确"
}
```

提交示例：

```bash
curl -X POST \
  "http://127.0.0.1:9001/api/v1/bvh/process" \
  -H "Content-Type: application/json" \
  -d '{
    "actionId":"action-42",
    "originalFileUrl":"https://minio.example.com/bucket/walk.bvh",
    "handleOptions":[1,2,3],
    "callbackUrl":"https://backend.example.com/callbacks/bvh"
  }'
```

## 处理结果回调

服务通过 `POST callbackUrl` 发起 `multipart/form-data` 请求，并在
`X-Callback-Token` 请求头携带回调鉴权 Token。

每个选中的处理选项完成后，会先发送一次进度回调：

```text
actionId=action-42
success=true
handleOption=2
optionStatus=completed
message=处理选项 2 完成
```

选项失败时 `success=false`、`optionStatus=failed`，不会上传 `file`。所有选项
完成后，再发送一次最终回调并携带处理后的 `file`；最终回调不携带
`handleOption`。

处理成功时包含：

```text
actionId=action-42
success=true
message=处理成功
file=<walk_processed.bvh 文件内容>
```

处理失败时不传 `file`：

```text
actionId=action-42
success=false
message=<具体失败原因>
```

如果回调请求失败，当前版本会记录错误日志，不会无限重试。

## 配置

复制 `.env.example` 为 `.env` 后按部署环境修改。

可配置项：

- `BVH_DOWNLOAD_TIMEOUT_SECONDS`：下载及回调超时秒数，默认 `30`。
- `BVH_MAX_FILE_SIZE_MB`：最大 BVH 文件大小，默认 `100`。
- `BVH_MINIO_ALLOWED_HOSTS`：MinIO 主机白名单，多个用逗号分隔。
- `BVH_CALLBACK_ALLOWED_HOSTS`：回调主机白名单，多个用逗号分隔。

白名单为空时允许任意 HTTP/HTTPS 主机，仅适合本地联调；生产环境必须同时配置 MinIO 和回调主机白名单。

## 当前任务执行方式

联调版本使用 FastAPI 进程内后台任务。服务重启时，尚未完成的任务会丢失。正式生产环境需要改用 Redis + Celery、RQ 或其他持久化任务队列。

## 测试与检查

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
