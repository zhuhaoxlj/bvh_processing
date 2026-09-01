# GPU Training Control API

本文档描述 GPU Training Control 当前提供的全部 HTTP 接口。该服务负责查询 GPU、管理动作 NPZ 文件，以及创建、停止、恢复和查询 `whole_body_tracking` 训练任务。

## 1. 基本信息

### 服务地址

本机默认地址：

```text
http://127.0.0.1:6666
```

当前公网代理地址：

```text
https://y7b4jaa2-x1b58667-6666.zj02restapi.gpufree.cn:8443
```

下文使用以下变量表示服务地址和 Token：

```bash
BASE_URL="https://y7b4jaa2-x1b58667-6666.zj02restapi.gpufree.cn:8443"
TOKEN="<GPU_CONTROL_API_TOKEN>"
```

服务同时提供 OpenAPI 交互文档：

```text
GET /docs
```

### 数据格式

除上传 NPZ 的接口使用 `multipart/form-data` 外，请求和响应均使用 JSON。

### 时间字段

`created_at`、`started_at`、`finished_at`、`updated_at` 和 `modified_at` 是 Unix 时间戳，单位为秒。尚未发生的时间为 `null`。

## 2. 认证

以下三个公共接口不需要认证：

```text
GET /health
GET /api/v1/gpus
GET /api/v1/gpus/simple
```

其他接口必须在请求头中携带 Bearer Token：

```http
Authorization: Bearer <GPU_CONTROL_API_TOKEN>
```

服务端 Token 配置在：

```text
/root/gpu_training_control/.env
```

对应环境变量：

```dotenv
GPU_CONTROL_API_TOKEN=<一个高强度随机Token>
```

后端调用示例：

```bash
curl -H "Authorization: Bearer $TOKEN" "$BASE_URL/api/v1/jobs"
```

Token **不能**通过 `?token=` 查询参数传递。缺少 Token 或 Token 错误时返回：

```http
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer
```

```json
{
  "detail": "invalid or missing bearer token"
}
```

如果服务端没有配置 `GPU_CONTROL_API_TOKEN`，受保护接口返回 `503`。

## 3. 接口总览

当前共提供 **14 个业务接口**。

| 分类 | 方法 | 路径 | 认证 | 用途 |
|---|---|---|---|---|
| 服务 | GET | `/health` | 否 | 健康检查 |
| GPU | GET | `/api/v1/gpus/simple` | 否 | 查询用于选择 GPU 的简化信息 |
| GPU | GET | `/api/v1/gpus` | 否 | 查询 GPU、显存、进程和预留详情 |
| 动作文件 | POST | `/api/v1/artifacts/motions` | 是 | 上传并校验动作 NPZ |
| 动作文件 | GET | `/api/v1/artifacts/motions` | 是 | 查询动作文件列表 |
| 动作文件 | GET | `/api/v1/artifacts/motions/{artifact_id}` | 是 | 查询单个动作文件 |
| 训练任务 | POST | `/api/v1/jobs` | 是 | 创建训练任务 |
| 训练任务 | GET | `/api/v1/jobs` | 是 | 查询所有训练任务及任务 ID |
| 训练任务 | GET | `/api/v1/jobs/{job_id}` | 是 | 查询单个任务详情 |
| 训练任务 | GET | `/api/v1/jobs/{job_id}/logs` | 是 | 查询任务控制台日志 |
| 训练任务 | GET | `/api/v1/jobs/{job_id}/loss` | 是 | 查询任务 TensorBoard loss 曲线 |
| 训练任务 | GET | `/api/v1/jobs/{job_id}/checkpoints` | 是 | 查询任务模型检查点 |
| 训练任务 | POST | `/api/v1/jobs/{job_id}/stop` | 是 | 停止正在运行的任务 |
| 训练任务 | POST | `/api/v1/jobs/{job_id}/resume` | 是 | 从检查点创建恢复训练任务 |

---

## 4. 服务接口

### 4.1 健康检查

```http
GET /health
```

无需认证。

请求示例：

```bash
curl "$BASE_URL/health"
```

成功响应，`200 OK`：

```json
{
  "status": "ok"
}
```

该接口只表示 HTTP 服务可以响应，不代表 GPU 或训练后端一定可用。

---

## 5. GPU 接口

### 5.1 查询简化 GPU 信息

```http
GET /api/v1/gpus/simple
```

无需认证。适合前端展示 GPU 选择列表。

请求示例：

```bash
curl "$BASE_URL/api/v1/gpus/simple"
```

成功响应，`200 OK`：

```json
{
  "gpus": [
    {
      "gpu": 0,
      "model": "NVIDIA GeForce RTX 4090",
      "memory_gib": 24.0,
      "available": true
    }
  ]
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `gpu` | integer | 物理 GPU 编号，创建任务时写为 `cuda:N` |
| `model` | string | GPU 型号 |
| `memory_gib` | number | 总显存，单位 GiB |
| `available` | boolean | 当前是否可由控制服务创建新任务 |

无法执行 `nvidia-smi` 时返回 `503`。

### 5.2 查询详细 GPU 信息

```http
GET /api/v1/gpus
```

无需认证。适合调度和排查 GPU 占用。

请求示例：

```bash
curl "$BASE_URL/api/v1/gpus"
```

成功响应，`200 OK`：

```json
{
  "gpu_count": 1,
  "available_count": 1,
  "gpus": [
    {
      "index": 0,
      "device": "cuda:0",
      "uuid": "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mib": 24564,
      "memory_used_mib": 1024,
      "memory_free_mib": 23540,
      "utilization_percent": 5,
      "temperature_celsius": 38,
      "power_draw_watts": 35.2,
      "power_limit_watts": 450.0,
      "processes": [],
      "status": "available",
      "available": true,
      "reserved_by_job": null
    }
  ]
}
```

GPU 状态：

| 状态 | 说明 |
|---|---|
| `available` | 没有计算进程且未被控制服务预留 |
| `reserved` | 已预留给某个控制服务训练任务 |
| `training` | 检测到不在当前预留记录中的 `whole_body_tracking` 训练进程 |
| `busy` | 存在其他 CUDA 计算进程 |

`processes` 中每项包含 `pid`、`used_memory_mib` 和 `name`。无法查询 GPU 时返回 `503`。

---

## 6. 动作 NPZ 接口

动作文件 ID 的格式为：

```text
motion_<32位小写十六进制字符>
```

### 6.1 上传并校验动作 NPZ

```http
POST /api/v1/artifacts/motions
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

表单字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | file | 是 | 扩展名必须为 `.npz` 的动作文件 |

请求示例：

```bash
curl -X POST "$BASE_URL/api/v1/artifacts/motions" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/motion.npz"
```

服务会校验以下数组：

```text
fps
joint_pos
joint_vel
body_pos_w
body_quat_w
body_lin_vel_w
body_ang_vel_w
```

主要形状要求：

| 数组 | 形状 |
|---|---|
| `joint_pos` | `[frames, 29]` |
| `joint_vel` | `[frames, 29]` |
| `body_pos_w` | `[frames, 30, 3]` |
| `body_quat_w` | `[frames, 30, 4]` |
| `body_lin_vel_w` | `[frames, 30, 3]` |
| `body_ang_vel_w` | `[frames, 30, 3]` |

所有数组必须是有限数值；`fps` 必须为正数；四元数必须归一化。

成功响应，`201 Created`：

```json
{
  "id": "motion_0123456789abcdef0123456789abcdef",
  "original_name": "dance.npz",
  "sha256": "<64位SHA-256>",
  "size_bytes": 12345678,
  "metadata": {
    "trainable": true,
    "fps": 50.0,
    "frames": 5000,
    "joint_count": 29,
    "body_count": 30,
    "duration_seconds": 100.0
  },
  "created_at": 1787906500.0
}
```

文件格式或数据校验失败返回 `400`；缺少文件字段返回 `422`。

### 6.2 查询动作文件列表

```http
GET /api/v1/artifacts/motions
Authorization: Bearer <token>
```

请求示例：

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "$BASE_URL/api/v1/artifacts/motions"
```

成功响应，`200 OK`：

```json
{
  "artifacts": [
    {
      "id": "motion_0123456789abcdef0123456789abcdef",
      "original_name": "dance.npz",
      "sha256": "<64位SHA-256>",
      "size_bytes": 12345678,
      "metadata": {
        "trainable": true,
        "fps": 50.0,
        "frames": 5000,
        "joint_count": 29,
        "body_count": 30,
        "duration_seconds": 100.0
      },
      "created_at": 1787906500.0
    }
  ]
}
```

按创建时间倒序排列，当前最多返回 100 条。服务器内部文件路径不会返回给调用方。

### 6.3 查询单个动作文件

```http
GET /api/v1/artifacts/motions/{artifact_id}
Authorization: Bearer <token>
```

请求示例：

```bash
ARTIFACT_ID="motion_0123456789abcdef0123456789abcdef"

curl -H "Authorization: Bearer $TOKEN" \
  "$BASE_URL/api/v1/artifacts/motions/$ARTIFACT_ID"
```

成功响应与上传接口的响应结构相同。动作文件不存在时返回：

```http
404 Not Found
```

```json
{
  "detail": "motion artifact does not exist"
}
```

---

## 7. 训练任务接口

训练任务 ID 的格式类似：

```text
job_<32位小写十六进制字符>
```

任务状态：

| 状态 | 说明 |
|---|---|
| `queued` | 已排队 |
| `starting` | 正在启动 |
| `running` | 正在训练 |
| `stopping` | 正在停止 |
| `stopped` | 已由用户停止 |
| `completed` | 正常完成 |
| `failed` | 启动或训练失败 |

任务详情中的常见字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 训练任务 ID |
| `artifact_id` | string | 使用的动作文件 ID |
| `parent_job_id` | string/null | 恢复训练时对应的原任务 ID |
| `status` | string | 当前任务状态 |
| `config` | object | 创建任务时使用的训练配置 |
| `run_name` | string | TensorBoard 和检查点使用的运行名称 |
| `session_name` | string | 执行训练的 tmux 会话名称 |
| `error` | string/null | 失败原因 |
| `exit_code` | integer/null | 训练进程退出码 |
| `created_at` | number | 创建时间 |
| `started_at` | number/null | 开始时间 |
| `finished_at` | number/null | 结束时间 |
| `updated_at` | number | 最近更新时间 |

### 7.1 创建训练任务

```http
POST /api/v1/jobs
Content-Type: application/json
Authorization: Bearer <token>
```

请求字段：

| 字段 | 类型 | 必填 | 默认值 | 约束或说明 |
|---|---|---|---|---|
| `artifact_id` | string | 是 | 无 | 必须是已上传且可训练的动作文件 ID |
| `devices` | string[] | 是 | 无 | 1–8 个，格式为 `cuda:N`，不能重复 |
| `num_envs` | integer | 否 | `18432` | 允许 `7168`、`12288`、`16384`、`18432`、`20480`、`22528`、`24576` |
| `max_iterations` | integer | 否 | `10000` | 1–100000 |
| `num_steps_per_env` | integer | 否 | `24` | 1–256 |
| `num_mini_batches` | integer | 否 | `4` | 1–128，并且必须整除 `num_envs × num_steps_per_env` |
| `num_learning_epochs` | integer | 否 | `5` | 1–20 |
| `learning_rate` | number | 否 | `0.001` | `0.000001`–`0.1` |
| `desired_kl` | number | 否 | `0.01` | `0.00001`–`1.0` |
| `save_interval` | integer | 否 | `500` | 1–100000 |
| `run_name` | string/null | 否 | 根据文件名生成 | 最多 80 个字符；服务会清理不安全字符并追加任务 ID |

请求示例：

```bash
curl -X POST "$BASE_URL/api/v1/jobs" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "artifact_id": "motion_0123456789abcdef0123456789abcdef",
    "devices": ["cuda:0"],
    "num_envs": 18432,
    "max_iterations": 10000,
    "num_steps_per_env": 24,
    "num_mini_batches": 4,
    "num_learning_epochs": 5,
    "learning_rate": 0.001,
    "desired_kl": 0.01,
    "save_interval": 500,
    "run_name": "dance"
  }'
```

成功响应，`202 Accepted`：

```json
{
  "id": "job_0123456789abcdef0123456789abcdef",
  "artifact_id": "motion_0123456789abcdef0123456789abcdef",
  "parent_job_id": null,
  "status": "running",
  "config": {
    "artifact_id": "motion_0123456789abcdef0123456789abcdef",
    "devices": ["cuda:0"],
    "num_envs": 18432,
    "max_iterations": 10000,
    "num_steps_per_env": 24,
    "num_mini_batches": 4,
    "num_learning_epochs": 5,
    "learning_rate": 0.001,
    "desired_kl": 0.01,
    "save_interval": 500,
    "run_name": "dance"
  },
  "run_name": "dance_job_01234567",
  "session_name": "gtc_0123456789ab",
  "error": null,
  "exit_code": null,
  "created_at": 1787906500.0,
  "started_at": 1787906501.0,
  "finished_at": null,
  "updated_at": 1787906501.0
}
```

常见错误：

- `409`：动作文件不存在、不可训练，或者所选 GPU 不可用。
- `422`：请求字段格式或取值不符合约束。
- `500`：训练进程启动失败。

### 7.2 查询所有训练任务

```http
GET /api/v1/jobs
Authorization: Bearer <token>
```

该接口用于先获取任务 ID，再查询日志、loss、检查点或任务详情。

请求示例：

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "$BASE_URL/api/v1/jobs"
```

成功响应，`200 OK`：

```json
{
  "jobs": [
    {
      "id": "job_0123456789abcdef0123456789abcdef",
      "artifact_id": "motion_0123456789abcdef0123456789abcdef",
      "parent_job_id": null,
      "status": "running",
      "config": {
        "artifact_id": "motion_0123456789abcdef0123456789abcdef",
        "devices": ["cuda:0"],
        "num_envs": 18432,
        "max_iterations": 10000
      },
      "run_name": "dance_job_01234567",
      "session_name": "gtc_0123456789ab",
      "error": null,
      "exit_code": null,
      "created_at": 1787906500.0,
      "started_at": 1787906501.0,
      "finished_at": null,
      "updated_at": 1787906501.0
    }
  ]
}
```

任务按创建时间倒序排列，当前最多返回最近 100 条。接口会先协调正在运行任务的最新状态。

### 7.3 查询单个训练任务

```http
GET /api/v1/jobs/{job_id}
Authorization: Bearer <token>
```

请求示例：

```bash
JOB_ID="job_0123456789abcdef0123456789abcdef"

curl -H "Authorization: Bearer $TOKEN" \
  "$BASE_URL/api/v1/jobs/$JOB_ID"
```

成功响应为单个任务对象，字段与“查询所有训练任务”中的任务对象相同。查询前会协调该任务的最新状态。

任务不存在时返回 `404`：

```json
{
  "detail": "training job does not exist"
}
```

### 7.4 查询训练日志

```http
GET /api/v1/jobs/{job_id}/logs
Authorization: Bearer <token>
```

返回训练进程控制台日志的末尾部分，当前最多读取约 128 KiB。

请求示例：

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "$BASE_URL/api/v1/jobs/$JOB_ID/logs"
```

成功响应，`200 OK`：

```json
{
  "job_id": "job_0123456789abcdef0123456789abcdef",
  "log": "GPU Training Control launch\n...训练日志...\n"
}
```

任务存在但日志文件尚未生成时，`log` 可能为空字符串。任务不存在时返回 `404`。

### 7.5 查询 loss 曲线

```http
GET /api/v1/jobs/{job_id}/loss
Authorization: Bearer <token>
```

从该任务的 TensorBoard event 文件中读取以下 scalar：

| 返回字段 | TensorBoard tag |
|---|---|
| `value_function` | `Loss/value_function` |
| `surrogate` | `Loss/surrogate` |
| `entropy` | `Loss/entropy` |

查询参数：

| 参数 | 类型 | 必填 | 默认值 | 约束 |
|---|---|---|---|---|
| `max_points` | integer | 否 | `2000` | 2–10000，每条曲线的最大返回点数 |

数据超过 `max_points` 时会等距采样，并保留首尾点。

请求示例：

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "$BASE_URL/api/v1/jobs/$JOB_ID/loss?max_points=500"
```

成功响应，`200 OK`：

```json
{
  "job_id": "job_0123456789abcdef0123456789abcdef",
  "run_directory": "2026-08-28_15-00-00_dance_job_01234567",
  "losses": {
    "value_function": [
      {"step": 0, "value": 0.143},
      {"step": 1, "value": 0.118}
    ],
    "surrogate": [
      {"step": 0, "value": -0.004}
    ],
    "entropy": [
      {"step": 0, "value": 12.31}
    ]
  }
}
```

某个 TensorBoard tag 不存在时，对应字段不会出现在 `losses` 中。

以下情况返回 `404`：

- 任务不存在。
- 该任务的 TensorBoard 运行目录尚未生成。
- TensorBoard event 文件暂时无法读取。
- 尚未记录任何受支持的 loss scalar。

`max_points` 超出范围或不是整数时返回 `422`。

### 7.6 查询模型检查点

```http
GET /api/v1/jobs/{job_id}/checkpoints
Authorization: Bearer <token>
```

请求示例：

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "$BASE_URL/api/v1/jobs/$JOB_ID/checkpoints"
```

成功响应，`200 OK`：

```json
{
  "job_id": "job_0123456789abcdef0123456789abcdef",
  "checkpoints": [
    {
      "run_directory": "2026-08-28_15-00-00_dance_job_01234567",
      "checkpoint_name": "model_2000.pt",
      "iteration": 2000,
      "size_bytes": 123456789,
      "modified_at": 1787907500.0
    }
  ]
}
```

检查点按迭代次数和修改时间倒序排列。任务存在但还没有检查点时返回空数组。任务不存在时返回 `404`。

### 7.7 停止训练任务

```http
POST /api/v1/jobs/{job_id}/stop
Authorization: Bearer <token>
```

只能停止状态为 `starting` 或 `running` 的任务。服务向任务的 tmux 会话发送 `Ctrl+C`，接口先返回 `stopping`，随后后台协调为 `stopped` 或 `failed`。

请求示例：

```bash
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  "$BASE_URL/api/v1/jobs/$JOB_ID/stop"
```

成功响应，`202 Accepted`：

```json
{
  "id": "job_0123456789abcdef0123456789abcdef",
  "status": "stopping",
  "artifact_id": "motion_0123456789abcdef0123456789abcdef",
  "parent_job_id": null,
  "config": {},
  "run_name": "dance_job_01234567",
  "session_name": "gtc_0123456789ab",
  "error": null,
  "exit_code": null,
  "created_at": 1787906500.0,
  "started_at": 1787906501.0,
  "finished_at": null,
  "updated_at": 1787906600.0
}
```

任务不存在、状态不允许停止或训练会话已结束时返回 `409`；发送停止指令失败时返回 `500`。

### 7.8 从检查点恢复训练

```http
POST /api/v1/jobs/{job_id}/resume
Content-Type: application/json
Authorization: Bearer <token>
```

该操作不会修改原任务，而是创建一个新任务。新任务的 `parent_job_id` 指向原任务 ID。

请求字段：

| 字段 | 类型 | 必填 | 默认值 | 约束或说明 |
|---|---|---|---|---|
| `run_directory` | string | 是 | 无 | 检查点所在的直接子目录名，不能包含 `/` 或 `\\` |
| `checkpoint_name` | string | 是 | 无 | 格式必须为 `model_<迭代次数>.pt` |
| `devices` | string[] | 是 | 无 | 1–8 个，格式为 `cuda:N`，不能重复 |
| `max_iterations` | integer | 否 | `10000` | 1–100000 |
| `num_envs` | integer/null | 否 | 沿用原任务 | 如提供，必须是创建任务支持的 `num_envs` 值之一 |

通常先调用检查点查询接口，直接使用返回的 `run_directory` 和 `checkpoint_name`。

请求示例：

```bash
curl -X POST "$BASE_URL/api/v1/jobs/$JOB_ID/resume" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "run_directory": "2026-08-28_15-00-00_dance_job_01234567",
    "checkpoint_name": "model_2000.pt",
    "devices": ["cuda:0"],
    "max_iterations": 10000,
    "num_envs": 18432
  }'
```

成功响应，`202 Accepted`，内容为新创建的任务对象：

```json
{
  "id": "job_fedcba9876543210fedcba9876543210",
  "artifact_id": "motion_0123456789abcdef0123456789abcdef",
  "parent_job_id": "job_0123456789abcdef0123456789abcdef",
  "status": "running",
  "config": {
    "artifact_id": "motion_0123456789abcdef0123456789abcdef",
    "devices": ["cuda:0"],
    "num_envs": 18432,
    "max_iterations": 10000,
    "resume": {
      "run_directory": "2026-08-28_15-00-00_dance_job_01234567",
      "checkpoint_name": "model_2000.pt"
    }
  },
  "run_name": "dance_job_01234567_resume_job_fedcba98",
  "session_name": "gtc_fedcba987654",
  "error": null,
  "exit_code": null,
  "created_at": 1787908500.0,
  "started_at": 1787908501.0,
  "finished_at": null,
  "updated_at": 1787908501.0
}
```

常见错误：

- `409`：原任务不存在、检查点不存在、GPU 不可用或恢复配置无效。
- `422`：请求字段格式或取值不符合约束。
- `500`：恢复训练进程启动失败。

---

## 8. 推荐调用流程

### 8.1 新建训练

```text
1. GET  /api/v1/gpus/simple
   查询可用 GPU

2. POST /api/v1/artifacts/motions
   上传 NPZ，保存响应中的 artifact_id

3. POST /api/v1/jobs
   使用 artifact_id 和 devices 创建任务，保存响应中的 job_id

4. GET  /api/v1/jobs/{job_id}
   查询任务状态

5. GET  /api/v1/jobs/{job_id}/logs
   查询控制台日志

6. GET  /api/v1/jobs/{job_id}/loss
   查询 loss 曲线

7. GET  /api/v1/jobs/{job_id}/checkpoints
   查询可用模型检查点
```

### 8.2 已有任务查询

```text
1. GET /api/v1/jobs
   获取最近任务及其 id

2. 使用选中的 job_id 查询：
   GET /api/v1/jobs/{job_id}
   GET /api/v1/jobs/{job_id}/logs
   GET /api/v1/jobs/{job_id}/loss
   GET /api/v1/jobs/{job_id}/checkpoints
```

### 8.3 恢复训练

```text
1. GET  /api/v1/jobs/{job_id}/checkpoints
2. 从响应中选择 run_directory 和 checkpoint_name
3. POST /api/v1/jobs/{job_id}/resume
4. 保存响应中新的 job_id；后续查询新任务，而不是原任务
```

## 9. 后端集成示例

### Python `httpx`

```python
import os

import httpx

BASE_URL = os.environ["GPU_TRAINING_CONTROL_URL"]
TOKEN = os.environ["GPU_TRAINING_CONTROL_TOKEN"]

client = httpx.Client(
    base_url=BASE_URL,
    headers={"Authorization": f"Bearer {TOKEN}"},
    timeout=30.0,
)

# 查询所有任务
response = client.get("/api/v1/jobs")
response.raise_for_status()
jobs = response.json()["jobs"]

if jobs:
    job_id = jobs[0]["id"]

    details = client.get(f"/api/v1/jobs/{job_id}").json()
    logs = client.get(f"/api/v1/jobs/{job_id}/logs").json()
    loss = client.get(
        f"/api/v1/jobs/{job_id}/loss",
        params={"max_points": 500},
    ).json()
```

### Node.js `fetch`

```javascript
const baseUrl = process.env.GPU_TRAINING_CONTROL_URL;
const token = process.env.GPU_TRAINING_CONTROL_TOKEN;

async function controlRequest(path, options = {}) {
  const response = await fetch(`${baseUrl}${path}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(options.headers ?? {}),
    },
  });

  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail ?? `HTTP ${response.status}`);
  }
  return body;
}

const { jobs } = await controlRequest("/api/v1/jobs");
if (jobs.length > 0) {
  const jobId = jobs[0].id;
  const details = await controlRequest(`/api/v1/jobs/${jobId}`);
  const logs = await controlRequest(`/api/v1/jobs/${jobId}/logs`);
  const loss = await controlRequest(
    `/api/v1/jobs/${jobId}/loss?max_points=500`,
  );
}
```

## 10. 通用错误响应

FastAPI 错误通常采用以下格式：

```json
{
  "detail": "错误说明"
}
```

常见状态码：

| 状态码 | 说明 |
|---|---|
| `200` | 查询成功 |
| `201` | 动作文件上传并创建成功 |
| `202` | 训练创建、停止或恢复请求已接受 |
| `400` | 上传文件内容或 NPZ 格式错误 |
| `401` | Bearer Token 缺失或错误 |
| `404` | 动作文件、训练任务、日志关联数据或 loss 数据不存在 |
| `409` | GPU 占用、任务状态冲突、检查点不存在或无法执行该操作 |
| `422` | JSON、路径参数、查询参数或字段校验失败 |
| `500` | 训练进程启动或控制失败 |
| `503` | Token 未配置，或者 GPU 查询服务不可用 |

字段校验失败的 `422` 响应会包含错误位置和原因，例如：

```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["query", "max_points"],
      "msg": "Input should be greater than or equal to 2",
      "input": "1",
      "ctx": {
        "ge": 2
      }
    }
  ]
}
```

## 11. 安全建议

- 只允许调用方后端保存 `GPU_CONTROL_API_TOKEN`。
- 浏览器前端调用自己的业务后端，由业务后端添加 `Authorization` 请求头。
- 不要把 Token 放在 URL 查询参数、前端静态代码、Git 仓库或日志中。
- 公网调用必须使用 HTTPS，并建议通过安全组、来源 IP 白名单或 VPN 进一步限制访问。
- 修改 `/root/gpu_training_control/.env` 中的 Token 后，需要重启控制服务。
