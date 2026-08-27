# Motion Inspector

本地 HHTools PKL 转换、NPZ 完整性检查、Isaac Sim MP4 预览与训练工具。

## 启动

在 `whole_body_tracking` 目录中运行：

```bash
./scripts/run_npz_preview_web.sh
```

也可以手动启动：

```bash
export VIRTUAL_ENV="$HOME/Project/01-RL/IsaacLab/env_isaaclab"
export PYTHONPATH="$PWD/source/whole_body_tracking"

"$VIRTUAL_ENV/bin/python" scripts/npz_preview_web.py
```

浏览器打开：<http://127.0.0.1:8765>

## 支持的动作文件

- `.npz`：直接执行现有完整性检查、预览和训练流程。
- HHTools `.pkl`：使用受限 pickle 解析器读取 NumPy 轨迹，按关节名称提取当前 G1 的 29 DoF，
  忽略手指等非训练关节，再调用 `scripts/csv_to_npz.py` 生成完整训练 NPZ。

PKL 的输入帧率从 `robot.sample_rate` 自动读取，输出统一为当前策略控制频率 50 Hz：

- 50 Hz PKL 保持 50 Hz 时间基准，只生成 FK、刚体姿态和速度字段；
- 120 Hz、59.94 Hz 等其他正数帧率会通过现有位置插值和四元数 SLERP 重采样到 50 Hz；
- 根四元数支持 `robot.root_quat_format` 为 `wxyz` 或 `xyzw`；
- PKL 不允许使用“快速训练通道”，必须完成安全解析、转换和 NPZ 完整性检查。

每个 PKL 任务目录会保留上传的原始 PKL、29 DoF 中间 CSV、转换日志和最终 50 Hz NPZ，
便于复查转换来源。PKL 转换默认使用 `cuda:0` 启动 Isaac Lab。

启动脚本不依赖 `uv`，会优先使用已经激活的 `VIRTUAL_ENV`，否则检查当前用户目录下常见的
Isaac Lab 环境位置。如果环境或 Python 安装在其他位置，可显式指定：

```bash
MOTION_PREVIEW_VENV="$HOME/Project/01-RL/IsaacLab/env_isaaclab" \
MOTION_PREVIEW_PYTHON="$HOME/Project/01-RL/IsaacLab/env_isaaclab/bin/python" \
  ./scripts/run_npz_preview_web.sh
```

可用参数：

```text
--host 127.0.0.1
--port 8765
--output-dir outputs/npz_preview_web
```

上传文件和渲染结果保存在 `outputs/npz_preview_web/<job-id>/`。服务默认只监听本机地址。

默认使用同步固定双机位：在动作第一帧按机器人的初始位置和朝向，布置正前方与正后方相机；
录制开始后两台相机的位置和朝向保持不变，不会跟随机器人移动或旋转。页面中的分辨率是单路相机分辨率，
例如“每路 960 × 720”会生成 1920 × 720 的左右分屏 MP4。也可切换回原有斜后方单机位。

镜头默认使用 18mm 超广角，水平视场角约 67°，适合移动范围较大的动作；也可以选择 24mm 广角或原来的
35mm 标准镜头。

## 从检查结果开始训练

如果训练日志出现 `Robot body index ... exceeds motion body count ...`，参见项目根目录的
[`NPZ_BODY_ORDER_TROUBLESHOOTING.md`](../../NPZ_BODY_ORDER_TROUBLESHOOTING.md)。不要通过跳过校验、补零或删除越界检查强行启动。

30-body、29-joint 且所有完整性检查通过的 NPZ 会显示训练按钮。默认设置遵循 `TRAINING_SOP.md`：

- task：`Tracking-Flat-G1-Wo-State-Estimation-v0`
- environments：表示每张 GPU 的并行环境数；默认 `18432`，双 4090 可选 `22528` 高利用率档或 `24576` 激进档，
  多卡总环境数为该值乘以 GPU 数量；使用激进档时需观察显存并在 OOM 后回退
- max iterations：`10000`
- steps per environment：默认 `24`，表示每次 PPO 更新前每个环境连续采集的控制步数
- mini-batches：页面默认 `16`（原始训练配置为 `4`），用于把超大双卡 rollout 切成更细的梯度更新
- learning epochs：默认 `5`，表示每轮 rollout 被完整遍历的次数
- learning rate：默认 `0.001`，继续配合 adaptive KL 调度
- desired KL：默认 `0.01`
- logger：TensorBoard
- device：可从本机可见 GPU 中多选；正在训练的设备会标记并禁用
- 单卡保持原有启动方式；多卡通过 `torch.distributed.run` 启动 RSL-RL 分布式训练，每个 rank 使用一张 GPU

后端会检查“每张 GPU 环境数 × steps per environment”能否被 mini-batches 整除，避免 RSL-RL 在切分
rollout storage 时失败。旧任务或直接调用后端但不提供这些字段时，仍使用项目原始 PPO 默认值：
`24 steps / 4 mini-batches / 5 epochs / 0.001 learning rate / 0.01 desired KL`。

训练在独立 tmux 会话中运行，关闭网页不会停止训练。页面会读取 `logs/manual/` 中的持久日志，提供
TensorBoard 入口，并可通过“停止训练”向 tmux 会话发送 `Ctrl+C`。启动任务时会一次锁定全部所选 GPU：同一张卡
不会同时启动训练或预览，未被选中的空闲 GPU 仍可用于其他任务。

训练停止、完成或失败后，状态区会扫描当前上传任务产生的 `model_N.pt`，默认选择迭代数最大的 checkpoint。
点击“继续训练”会加载该 checkpoint 的模型、优化器和归一化状态，在新的 run 目录中追加训练；表单中的迭代数
表示本次要追加的迭代数。旧 run 和已有 checkpoint 不会被覆盖，也可以在下拉框中选择更早的恢复点。

页面顶部的“历史训练任务”会按 `logs/rsl_rl/g1_flat/` 下的每个时间戳 Run 单独展示，不依赖浏览器
localStorage。Run 的 `params/env.yaml` 若能关联到 `outputs/npz_preview_web/` 中的上传动作，就会标记为“可续训”；
打开后默认选择该 Run 自己的最新 checkpoint，而不是同一动作其他 Run 的 checkpoint。外部动作和 smoke Run 也会
显示，但由于缺少网页上传任务上下文，会标记为不可在网页续训。

如果 NPZ 已在之前完成检查，可以在拖放区开启“快速训练通道”。该模式只上传和保存文件，不会用 NumPy
打开、解压或检查任何数组，并直接开放训练配置；为避免误解，页面会标记 `VALIDATION SKIPPED`，禁用预览功能，
开始训练前也会再次显示未检查警告。


```bash
export VIRTUAL_ENV="$HOME/Project/01-RL/IsaacLab/env_isaaclab"
export PYTHONPATH="$PWD/source/whole_body_tracking"

"$VIRTUAL_ENV/bin/python" scripts/npz_preview_web.py --port 8766
```
