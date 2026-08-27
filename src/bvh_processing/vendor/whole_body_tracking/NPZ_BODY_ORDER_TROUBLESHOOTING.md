# NPZ 与机器人 Body 顺序不一致排障手册

本文档处理以下典型问题：同一个动作 NPZ 在一台机器可以训练，复制到另一台机器后，在 Isaac Sim 环境初始化阶段失败。

典型错误：

```text
ValueError: Robot body index 34 exceeds motion body count 30.
Regenerate the NPZ with the same robot model/body ordering used for training.
```

这不是显存不足，也不是环境数过大。它表示：训练进程加载出的机器人刚体数量或顺序，与 NPZ 保存动作时使用的刚体数量或顺序不同。

## 1. 为什么 body 顺序必须一致

NPZ 包含以下刚体数组：

```text
body_pos_w
body_quat_w
body_lin_vel_w
body_ang_vel_w
```

它们的第二维是机器人刚体索引。例如：

```text
body_pos_w[frame, body_index, xyz]
```

训练时，`MotionCommand` 根据机器人 articulation 的 `body_names` 查找 pelvis、膝、脚踝、torso、手腕等刚体的索引，再用这些索引读取 NPZ。读取结果参与：

- motion observation；
- 身体位置、姿态、线速度和角速度 reward；
- anchor/body 偏差 termination；
- 训练指标统计。

如果顺序不一致，轻则索引越界并立即停止，重则把一个部位的参考数据错误地用于另一个部位，导致 reward 和 termination 失真。不要删除越界检查，也不要通过补零、复制 body 或截断数组绕过错误。

## 2. 当前标准环境

当前项目的 G1 动作格式应满足：

```text
joint count: 29
body count:  30
```

正确合并固定关节后，本项目关注的 body 索引为：

```text
pelvis:                    0
left_hip_roll_link:        4
right_hip_roll_link:       5
torso_link:                9
left_knee_link:           10
right_knee_link:          11
left_shoulder_roll_link:  16
right_shoulder_roll_link: 17
left_ankle_roll_link:     18
right_ankle_roll_link:    19
left_elbow_link:          22
right_elbow_link:         23
left_wrist_yaw_link:      28
right_wrist_yaw_link:     29
```

如果日志中手腕索引是 `33/34`，或生成的 NPZ 是 35/37 bodies，通常意味着 URDF importer 没有合并带质量/惯量声明的固定关节。

## 3. 先建立最小复现

不要直接用 7168 environments 重试。先使用 1 个环境和 1 次迭代，减少初始化时间和 GPU 占用。

设置路径：

```bash
project_root="$HOME/whole_body_tracking"
isaaclab_root="$HOME/Project/01-RL/IsaacLab"
isaaclab_python="$isaaclab_root/env_isaaclab/bin/python"
motion_path="/absolute/path/to/motion.npz"
```

运行 smoke test：

```bash
cd "$project_root"

PYTHONPATH="$project_root/source/whole_body_tracking${PYTHONPATH:+:$PYTHONPATH}" \
HYDRA_FULL_ERROR=1 \
  "$isaaclab_python" scripts/rsl_rl/train.py \
    --task Tracking-Flat-G1-Wo-State-Estimation-v0 \
    --motion_file "$motion_path" \
    --num_envs 1 \
    --max_iterations 1 \
    --run_name body_order_smoke_test \
    --logger tensorboard \
    --headless \
    --device cuda:0
```

该命令必须能复现正式训练的同一错误。修复后仍运行完全相同的命令，并确认出现：

```text
Learning iteration 0/1
```

## 4. 检查 NPZ，而不是根据文件名猜测

完整读取并打印 shape：

```bash
"$isaaclab_python" -c '
import sys
import numpy as np

motion_path = sys.argv[1]
data = np.load(motion_path, allow_pickle=False)
for name in data.files:
    array = data[name]
    print(name, array.shape, array.dtype, "finite=", bool(np.isfinite(array).all()))
print("joint_count=", data["joint_pos"].shape[1])
print("body_count=", data["body_pos_w"].shape[1])
' "$motion_path"
```

计算文件哈希：

```bash
sha256sum "$motion_path"
```

如果问题发生在两台机器，必须比较两端 SHA-256。哈希不同，先解决文件选择或传输问题，不要继续修改训练环境。

## 5. 比较项目代码和机器人资产

两台机器分别执行：

```bash
cd "$project_root"

git rev-parse HEAD
git status --short

sha256sum \
  source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf \
  source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/robot.xacro
```

依次判断：

1. 项目 commit 是否相同；
2. 工作区是否存在影响 G1 配置或资产的修改；
3. `main.urdf` 和 `robot.xacro` 哈希是否相同；
4. 训练 task 是否都是 `Tracking-Flat-G1-Wo-State-Estimation-v0`。

这些内容不一致时，应先同步项目版本。不要在两个不同机器人模型之间强行复用 NPZ。

## 6. 比较 Isaac Lab 和 URDF importer

检查 Isaac Lab：

```bash
git -C "$isaaclab_root" rev-parse HEAD
git -C "$isaaclab_root" status --short
```

检查转换器是否包含兼容逻辑：

```bash
rg -n '2\.4\.31|set_merge_fixed_ignore_inertia|set_merge_fixed_joints' \
  "$isaaclab_root/source/isaaclab/isaaclab/sim/converters/urdf_converter.py"
```

检查 Isaac Sim 自带 importer 版本：

```bash
rg -n '^version' \
  "$isaaclab_root/env_isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf/config/extension.toml"
```

Isaac Sim 5.1 改变了固定关节合并行为。仅设置 `merge_fixed_joints=True` 可能仍然保留带质量/惯量的固定 link，从而增加 articulation body 数量。

已知可用的兼容逻辑包含两部分：

```python
manager = omni.kit.app.get_app().get_extension_manager()
if not manager.is_extension_enabled("isaacsim.asset.importer.urdf-2.4.31"):
    manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf-2.4.31", True)
```

以及：

```python
import_config.set_merge_fixed_joints(self.cfg.merge_fixed_joints)
if hasattr(import_config, "set_merge_fixed_ignore_inertia"):
    import_config.set_merge_fixed_ignore_inertia(self.cfg.merge_fixed_joints)
```

启用后，扩展通常缓存在：

```text
~/.local/share/ov/data/exts/v2/isaacsim.asset.importer.urdf-2.4.31+*/
```

## 7. 修复顺序

推荐按以下顺序处理：

1. 确认 NPZ 哈希相同；
2. 确认项目 commit、task 和 URDF 哈希相同；
3. 确认问题只发生在 Isaac Lab / importer 版本不同的机器；
4. 优先将目标机器的 Isaac Lab 对齐到已经验证可用的版本；
5. 如果暂时不能升级，再回移第 6 节所示的上游兼容逻辑；
6. 重新运行 1 environment / 1 iteration smoke test；
7. smoke test 通过后再恢复正式环境数。

修改 Isaac Lab 前必须先执行：

```bash
git -C "$isaaclab_root" status --short
git -C "$isaaclab_root" diff -- \
  source/isaaclab/isaaclab/sim/converters/urdf_converter.py
```

如果目标文件已有其他未提交修改，不要直接覆盖；先保存 patch 或请维护者合并变更。

## 8. 修复后的验收标准

smoke test 需要同时满足：

1. NPZ 是 29 joints、30 bodies；
2. 日志中的最大 body 索引不超过 29；
3. `right_wrist_yaw_link` 为 29；
4. 不再出现 `exceeds motion body count`；
5. 环境、observation、reward 和 termination manager 完成初始化；
6. 至少完成 `Learning iteration 0/1`；
7. 进程退出码为 0。

快速检查日志：

```bash
rg 'Motion body indexes|exceeds motion body count|Learning iteration|Traceback|Error executing job' \
  /path/to/training.log
```

正式训练开始后，再检查：

```bash
nvidia-smi
tail -f /path/to/training.log
```

## 9. 禁止的处理方式

不要采用以下“看似能跑”的修复：

- 删除 `commands.py` 中的越界检查；
- 把 30-body NPZ 补零成 35/37 bodies；
- 截断 35/37-body NPZ；
- 手工把腕部索引从 34 改为 29，但不修复完整 body ordering；
- 仅根据 `_zhrm`、`_local` 等文件名判断兼容性；
- 未做 smoke test 就直接启动数千 environments。

这些方法可能让训练启动，但会把错误 body 的参考姿态和速度送入 reward，所得 checkpoint 不可信。

## 10. 预防措施

1. 在训练记录中保存项目 commit、Isaac Lab commit、Isaac Sim/importer 版本和 NPZ SHA-256；
2. 生成 NPZ 与训练尽量使用同一套锁定环境；
3. 新机器部署后，先用固定 NPZ 跑 1 environment / 1 iteration；
4. 看到 35/37 bodies 时立即停止，先检查固定关节合并行为；
5. 长期应在 NPZ 中加入明确的 `body_names` 元数据，让加载器按名称映射，而不是隐式依赖数组顺序；
6. Isaac Lab 升级后重新运行本手册的 smoke test，再开放正式训练。

## 11. 本次问题的参考结论

本次两端比较结果：

```text
NPZ SHA-256:      相同
项目 commit:      相同
G1 URDF SHA-256:  相同
本机 Isaac Lab:   d0554cc
远端 Isaac Lab:   3c6e67b
失败 importer:    2.4.30
修复 importer:    2.4.31
失败最大索引:     34
修复最大索引:     29
```

修复后，同一 NPZ 已在远端通过 1 environment / 1 iteration smoke test。
