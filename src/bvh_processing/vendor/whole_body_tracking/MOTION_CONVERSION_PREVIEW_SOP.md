# CSV 转 NPZ 与 MP4 预览 SOP

本文档说明如何在 `/home/zh/whole_body_tracking` 中：

1. 使用服务器当前 Isaac Lab 环境，将 G1 广义坐标 CSV 转换为与训练机器人刚体顺序一致的 NPZ；
2. 检查 NPZ 是否完整、刚体数量是否与当前训练环境的 G1 模型一致；
3. 将 NPZ 回放并渲染为 MP4 预览视频；
4. 判断 Isaac Sim 是仍在工作、卡在退出阶段，还是已经失败。

本文档中的 Python 命令统一通过 `uv` 管理和执行。

## 1. 环境与路径

项目目录：

```text
/home/zh/whole_body_tracking
```

Isaac Lab 虚拟环境：

```text
/home/zh/Project/01-RL/IsaacLab/env_isaaclab
```

`uv`：

```text
/home/zh/.local/bin/uv
```

进入项目目录：

```bash
cd /home/zh/whole_body_tracking
```

设置后续命令需要的环境变量：

```bash
export VIRTUAL_ENV=/home/zh/Project/01-RL/IsaacLab/env_isaaclab
export PYTHONPATH=/home/zh/whole_body_tracking/source/whole_body_tracking
```

所有 Python 命令使用：

```bash
/home/zh/.local/bin/uv run --active --no-sync python ...
```

- `--active`：使用 `VIRTUAL_ENV` 指向的 Isaac Lab 环境；
- `--no-sync`：不改动环境，也不尝试重新安装项目依赖。

不要直接在项目根目录运行未指定 `VIRTUAL_ENV` 的 `uv run`。否则 `uv` 可能创建新的 `.venv`，而该环境通常没有 Isaac Lab、Isaac Sim 和 NumPy。

## 2. 转换前检查 CSV

本次示例输入：

```text
/home/zh/whole_body_tracking/motions/Take_007_054_Skeleton7_0_g1_retarget.csv
```

确认文件存在并查看行数：

```bash
ls -lh /home/zh/whole_body_tracking/motions/Take_007_054_Skeleton7_0_g1_retarget.csv
wc -l /home/zh/whole_body_tracking/motions/Take_007_054_Skeleton7_0_g1_retarget.csv
```

CSV 每行应包含：

```text
root xyz + root quaternion xyzw + 29 个 G1 关节角
```

即每行通常应有 36 列。可通过 `uv` 检查：

```bash
VIRTUAL_ENV=/home/zh/Project/01-RL/IsaacLab/env_isaaclab \
  /home/zh/.local/bin/uv run --active --no-sync python -c '
import numpy as np

motion_csv = np.loadtxt(
    "/home/zh/whole_body_tracking/motions/Take_007_054_Skeleton7_0_g1_retarget.csv",
    delimiter=",",
)
print("shape:", motion_csv.shape)
print("all finite:", bool(np.isfinite(motion_csv).all()))
'
```

转换前必须确认输入帧率。当前 `Take_007_054_Skeleton7_0_g1_retarget.csv` 按 120 Hz 处理。错误的输入帧率会使动作速度和时长整体失真。

## 3. CSV 转换为环境兼容 NPZ

转换脚本：

```text
scripts/csv_to_npz.py
```

推荐为服务器训练环境生成的文件增加 `_zhrm` 后缀，保留原始文件：

```text
Take_007_054_Skeleton7_0_g1_retarget_zhrm.npz
```

执行转换：

```bash
cd /home/zh/whole_body_tracking

VIRTUAL_ENV=/home/zh/Project/01-RL/IsaacLab/env_isaaclab \
PYTHONPATH=/home/zh/whole_body_tracking/source/whole_body_tracking \
  /home/zh/.local/bin/uv run --active --no-sync \
  python scripts/csv_to_npz.py \
    --input_file /home/zh/whole_body_tracking/motions/Take_007_054_Skeleton7_0_g1_retarget.csv \
    --input_fps 120 \
    --output_name Take_007_054_Skeleton7_0_g1_retarget_zhrm \
    --output_fps 50 \
    --save_to /home/zh/whole_body_tracking/motions/Take_007_054_Skeleton7_0_g1_retarget_zhrm.npz \
    --no_wandb \
    --headless \
    --device cuda:0 \
    --shutdown_timeout 30
```

参数说明：

- `--input_file`：输入 CSV；
- `--input_fps 120`：CSV 的真实采样率；
- `--output_fps 50`：训练参考动作的输出采样率；
- `--save_to`：本地 NPZ 输出路径；
- `--no_wandb`：不上传 WandB；
- `--headless`：无窗口运行；
- `--device cuda:0`：明确使用 GPU 0。
- `--shutdown_timeout 30`：NPZ 写完后，若 Isaac Sim 插件关闭超过 30 秒，自动结束残留进程；设为 `0` 可禁用。

不建议对单进程 Isaac Sim 转换设置 `CUDA_VISIBLE_DEVICES=0`。Omniverse 会同时进行 CUDA 和 Vulkan 设备枚举，设置该变量可能产生设备编号不一致或 `CUDA being in bad state` 警告。优先使用：

```text
--device cuda:0
```

### 转换原理

该脚本不是简单地把 CSV 压缩成 NPZ，而是：

1. 按输入和输出帧率插值根节点、四元数和关节角；
2. 计算根节点和关节速度；
3. 在当前 Isaac Lab 环境中加载 `G1_CYLINDER_CFG`；
4. 将每帧根节点和 29 个关节状态写入机器人；
5. 从当前机器人模型读取所有刚体的位置、姿态和速度；
6. 保存为训练可直接读取的 NPZ。

因此，NPZ 中的刚体数量和顺序会与生成它的服务器训练环境一致。不要通过补零、复制刚体或手工改索引把旧 NPZ 伪造成兼容格式。

## 4. 验证 NPZ 完整性

文件出现后，不应仅根据文件名判断转换完成。使用 NumPy 完整加载所有数组：

```bash
VIRTUAL_ENV=/home/zh/Project/01-RL/IsaacLab/env_isaaclab \
  /home/zh/.local/bin/uv run --active --no-sync python -c '
import numpy as np

motion_path = "/home/zh/whole_body_tracking/motions/Take_007_054_Skeleton7_0_g1_retarget_zhrm.npz"
motion_data = np.load(motion_path)

print("keys:", motion_data.files)
for array_name in motion_data.files:
    array = motion_data[array_name]
    print(
        array_name,
        "shape=", array.shape,
        "dtype=", array.dtype,
        "finite=", bool(np.isfinite(array).all()),
    )

print("fps:", motion_data["fps"].tolist())
print("frames:", motion_data["joint_pos"].shape[0])
print("joint count:", motion_data["joint_pos"].shape[1])
print("body count:", motion_data["body_pos_w"].shape[1])
'
```

在已经修复固定关节合并行为、并与当前本机训练环境一致的 30-body 环境中，正确结果应为：

```text
fps:             (1,)        -> [50]
joint_pos:       (392, 29)
joint_vel:       (392, 29)
body_pos_w:      (392, 30, 3)
body_quat_w:     (392, 30, 4)
body_lin_vel_w:  (392, 30, 3)
body_ang_vel_w:  (392, 30, 3)
all finite:      True
```

训练前至少需要满足：

1. 所有标准键都存在；
2. 所有数组帧数一致；
3. 关节数量为 29；
4. 刚体数量与生成 NPZ、训练所用的 `G1_CYLINDER_CFG` 一致；当前标准为 30。若出现 35/37，先按 `NPZ_BODY_ORDER_TROUBLESHOOTING.md` 检查 Isaac Lab 和 URDF importer；
5. 所有数组无 NaN 和 Inf；
6. 四元数范数接近 1。

四元数检查：

```bash
VIRTUAL_ENV=/home/zh/Project/01-RL/IsaacLab/env_isaaclab \
  /home/zh/.local/bin/uv run --active --no-sync python -c '
import numpy as np

motion_data = np.load(
    "/home/zh/whole_body_tracking/motions/Take_007_054_Skeleton7_0_g1_retarget_zhrm.npz"
)
quaternion_norms = np.linalg.norm(motion_data["body_quat_w"], axis=-1)
print("quaternion norm min:", float(quaternion_norms.min()))
print("quaternion norm max:", float(quaternion_norms.max()))
'
```

## 5. NPZ 渲染为 MP4

回放与录制脚本：

```text
scripts/replay_npz.py
```

录制完整动作：

```bash
cd /home/zh/whole_body_tracking

VIRTUAL_ENV=/home/zh/Project/01-RL/IsaacLab/env_isaaclab \
PYTHONPATH=/home/zh/whole_body_tracking/source/whole_body_tracking \
  /home/zh/.local/bin/uv run --active --no-sync \
  python scripts/replay_npz.py \
    --motion_file /home/zh/whole_body_tracking/motions/Take_007_054_Skeleton7_0_g1_retarget_zhrm.npz \
    --output_video /home/zh/whole_body_tracking/motions/Take_007_054_Skeleton7_0_g1_retarget_zhrm_preview.mp4 \
    --video_width 960 \
    --video_height 720 \
    --enable_cameras \
    --headless \
    --device cuda:0 \
    --shutdown_timeout 30
```

关键参数：

- `--motion_file`：已验证的 NPZ；
- `--output_video`：MP4 输出路径；
- `--video_width`、`--video_height`：视频分辨率；
- `--enable_cameras`：启动 Isaac Sim 相机传感器，录制时必须提供；
- `--headless`：无需桌面窗口；
- `--device cuda:0`：使用 GPU 0 渲染。
- `--shutdown_timeout 30`：视频 writer 完整关闭后，若 Isaac Sim 插件关闭超过 30 秒，自动结束残留进程。

脚本会：

1. 加载同一个 `G1_CYLINDER_CFG`；
2. 逐帧写入根节点、关节位置和速度；
3. 让相机跟随机器人根节点；
4. 读取 RGB 帧；
5. 使用 H.264 编码完整动作周期；
6. 在最后一帧关闭视频写入器。

目标视频参数应为：

```text
分辨率：960 x 720
帧率：  50 FPS
帧数：  392
时长：  约 7.84 秒
编码：  H.264 / MP4
```

## 6. 验证 MP4

确认文件存在且非空：

```bash
ls -lh /home/zh/whole_body_tracking/motions/Take_007_054_Skeleton7_0_g1_retarget_zhrm_preview.mp4
```

如果系统安装了 `ffprobe`，检查编码、分辨率、帧率和时长：

```bash
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames \
  -show_entries format=duration,size \
  -of default=noprint_wrappers=1 \
  /home/zh/whole_body_tracking/motions/Take_007_054_Skeleton7_0_g1_retarget_zhrm_preview.mp4
```

也可以用 `uv` 和 `imageio`/`opencv` 所在的现有环境读取首帧，但 `ffprobe` 更适合验证 MP4 容器是否正确收尾。

## 7. 监控运行状态

查看转换或录制进程：

```bash
pgrep -af 'scripts/csv_to_npz.py|scripts/replay_npz.py'
```

查看主进程状态和 CPU 使用：

```bash
ps -o pid,stat,etime,pcpu,pmem,wchan:32 -p <PID>
```

常见状态解释：

- `R`：正在 CPU 上执行；
- `S`：等待事件，Isaac Sim 多线程程序中很常见；
- 高 CPU 使用率：通常表示仍在导入、渲染或处理帧；
- `Z`：僵尸进程，需要检查父进程；
- 长时间 0% CPU、无 GPU 占用、无日志变化：可能卡死。

查看 GPU 进程：

```bash
nvidia-smi --query-compute-apps=pid,used_memory \
  --format=csv,noheader,nounits
```

Isaac Sim 启动时通常会输出大量警告，包括：

```text
GLFW initialization failed
failed to open the default display
Unresolved reference prim path
deprecated extension
```

在 `--headless` 模式下，这些警告不一定表示失败。判断是否真正失败时，应优先检查：

1. 是否出现 Python `Traceback`；
2. 进程是否退出且返回非零状态；
3. 输出文件是否可被 NumPy 或 ffprobe 完整读取；
4. CPU/GPU 是否仍在工作；
5. 日志是否继续增长。

## 8. 文件已生成但进程仍未退出

转换脚本可能已经执行：

```python
np.savez(...)
```

但随后卡在 Isaac Sim 插件关闭阶段。此时文件可能已经完整可用。

处理顺序：

1. 查看文件修改时间和大小：

   ```bash
   stat --format='%n size=%s bytes modified=%y' \
     /home/zh/whole_body_tracking/motions/Take_007_054_Skeleton7_0_g1_retarget_zhrm.npz
   ```

2. 使用第 4 节的 NumPy 命令完整加载全部数组；
3. 确认 body 数与当前 `G1_CYLINDER_CFG` 一致、29 joints、所有数据 finite；当前标准为 30 bodies，若出现 35/37 应停止并检查固定关节合并行为；
4. 确认文件修改时间不再变化；
5. 只有在以上检查全部通过后，才可以结束残留转换进程。

优先向前台进程发送 `Ctrl+C`。如果它不是前台进程，可以先确认 PID，再发送 `SIGINT`：

```bash
kill -INT <PID>
```

等待一段时间后仍未退出，才考虑：

```bash
kill -TERM <PID>
```

不要在没有验证输出文件完整性的情况下直接终止转换进程。

当前 `scripts/csv_to_npz.py` 默认带有关闭 watchdog：`np.savez` 完成并返回主流程后，先正常调用
`simulation_app.close()`；若关闭阶段超过 `--shutdown_timeout`（默认 30 秒），脚本打印警告并以成功状态退出。
操作系统会回收该进程的 CUDA、Vulkan 和插件资源。该机制只处理“文件已完整保存后的关闭卡死”，不会掩盖转换阶段的异常。

`scripts/replay_npz.py` 使用相同机制，但 watchdog 只在 `video_writer.close()` 完成、视频已打印保存成功后启动，
因此不会截断 MP4 或导致缺少容器尾部索引。

## 9. MP4 在录制结束前不可播放

MP4/H.264 写入器通常在关闭时才写入或完成容器索引。因此：

- 录制过程中，目标 MP4 可能暂时不存在；
- 文件出现后也可能暂时无法播放；
- 只有脚本打印 `Preview video saved` 并关闭 writer 后，才应将其视为最终文件；
- 如果进程被强制终止，MP4 可能缺少尾部索引而无法播放。

因此，除非已确认渲染进程卡死，不要在录制过程中终止 `replay_npz.py`。

## 10. 建议的标准命名

原始或重定向 CSV：

```text
<motion>_g1_retarget.csv
```

服务器环境兼容 NPZ：

```text
<motion>_g1_retarget_zhrm.npz
```

预览视频：

```text
<motion>_g1_retarget_zhrm_preview.mp4
```

文件名后缀不能证明兼容性。正式训练前必须实际检查 NPZ 为 29 joints、30 bodies，并运行 1 environment / 1 iteration smoke test；详细步骤见 `NPZ_BODY_ORDER_TROUBLESHOOTING.md`。

## 11. 当前文件

当前已生成并验证的 NPZ：

```text
/home/zh/whole_body_tracking/motions/Take_007_054_Skeleton7_0_g1_retarget_zhrm.npz
```

当前预览视频目标路径：

```text
/home/zh/whole_body_tracking/motions/Take_007_054_Skeleton7_0_g1_retarget_zhrm_preview.mp4
```
