# G1 23 DoF User Control 切换测试说明

本文介绍 [`tools/g1_userctrl_23dof_switch.cpp`](tools/g1_userctrl_23dof_switch.cpp) 的用途、工作流程、机器人端编译部署方式、运行方法和安全注意事项。

该程序只用于验证 Unitree G1 在公版 `WALKRUN` 与 SDK2 User Control 之间的控制权切换，不包含 BeyondMimic、行走策略或舞蹈模型推理。

## 1. 程序用途

程序执行以下流程：

```text
公版 WALKRUN（fsm_id=801、fsm_mode=0）
    ↓
读取 rt/lowstate，并记录当前23个有效关节角
    ↓
以50 Hz预发布 rt/user_lowcmd 保持命令
    ↓
SwitchToUserCtrl()
    ↓
在指定时间内保持切换瞬间的关节姿态
    ↓
SwitchToInternalCtrl(WALKRUN)
    ↓
确认公版 WALKRUN 已重新接管
```

程序不会：

- 发送行走速度；
- 播放 ONNX 策略；
- 改变目标姿态；
- 控制23 DoF机器上不存在的6个关节；
- 在 `--dry-run` 模式下发布控制命令或切换控制权。

## 2. 23 DoF关节映射

程序只控制以下23个电机索引：

```text
0-12    双腿12个关节 + waist_yaw
15-19   左臂5个关节
22-26   右臂5个关节
```

以下6个29 DoF扩展关节在23 DoF机器人上不可用，程序对其保持零位置、零速度、零刚度、零阻尼和零前馈力矩：

| 电机索引 | 关节 |
| --- | --- |
| 13 | `waist_roll` |
| 14 | `waist_pitch` |
| 20 | `left_wrist_pitch` |
| 21 | `left_wrist_yaw` |
| 27 | `right_wrist_pitch` |
| 28 | `right_wrist_yaw` |

23个有效关节的保持参数为：

```text
q   = 切换前读取的当前关节角
dq  = 0
kp  = 20
kd  = 1
tau = 0
```

## 3. 前置条件

机器人端需要：

- G1 23 DoF机器人；
- 支持 `SwitchToUserCtrl()` 和 `SwitchToInternalCtrl()` 的新版 `unitree_sdk2`；
- SDK路径：`/home/unitree/zh/unitree_sdk2`；
- DDS控制网卡：`eth0`；
- 能够收到 `rt/lowstate`；
- 机器人已进入本机固件对应的 `WALKRUN`：`fsm_id=801`、`fsm_mode=0`；
- 没有其他程序同时发布低层控制命令。

> `fsm_id=801`、`fsm_mode=0` 是当前测试机器人上的实机结果，不应未经查询直接套用到其他固件版本。

## 4. 上传和持久化安装

### 4.1 从开发机上传源码

在本仓库根目录执行：

```bash
scp tools/g1_userctrl_23dof_switch.cpp \
  unitree@192.168.123.164:/home/unitree/bin/g1_userctrl_23dof_switch.cpp
```

不要把正式测试程序只放在 `/tmp`。机器人重启或系统清理后，`/tmp` 中的源码和二进制可能消失。

### 4.2 在机器人上编译

登录机器人：

```bash
ssh unitree@192.168.123.164
```

创建持久目录：

```bash
mkdir -p /home/unitree/bin
```

使用机器人上的新版 SDK 编译：

```bash
g++ -std=c++17 -O2 -Wall -Wextra \
  -I/home/unitree/zh/unitree_sdk2/include \
  -I/home/unitree/zh/unitree_sdk2/thirdparty/include \
  -I/home/unitree/zh/unitree_sdk2/thirdparty/include/ddscxx \
  /home/unitree/bin/g1_userctrl_23dof_switch.cpp \
  -L/home/unitree/zh/unitree_sdk2/lib/aarch64 \
  -L/home/unitree/zh/unitree_sdk2/thirdparty/lib/aarch64 \
  -Wl,-rpath,/home/unitree/zh/unitree_sdk2/lib/aarch64:/home/unitree/zh/unitree_sdk2/thirdparty/lib/aarch64 \
  -lunitree_sdk2 -lddscxx -lddsc -lpthread -ldl \
  -o /home/unitree/bin/g1_userctrl_23dof_switch
```

设置执行权限：

```bash
chmod 755 /home/unitree/bin/g1_userctrl_23dof_switch
```

### 4.3 验证安装

```bash
ls -l /home/unitree/bin/g1_userctrl_23dof_switch
ldd /home/unitree/bin/g1_userctrl_23dof_switch | grep 'not found'
sha256sum /home/unitree/bin/g1_userctrl_23dof_switch.cpp
```

`ldd` 命令没有输出 `not found` 才表示运行库完整。

## 5. 运行方法

命令格式：

```text
g1_userctrl_23dof_switch <network_interface> [duration_sec] [--dry-run]
```

### 5.1 先执行无控制 dry-run

```bash
/home/unitree/bin/g1_userctrl_23dof_switch eth0 5 --dry-run
```

正常输出：

```text
Preflight OK: fsm_id=801, fsm_mode=0, lowstate received, 23 joint targets finite
Dry run complete: no rt/user_lowcmd published and no control switch requested
```

`--dry-run` 只检查：

- FSM状态；
- `rt/lowstate` 是否可用；
- 23个有效关节角是否为有限合理数值；
- DDS网卡和 SDK API 是否可用。

它不会发布 `rt/user_lowcmd`，也不会调用 `SwitchToUserCtrl()`。

### 5.2 真实切换并保持5秒

```bash
/home/unitree/bin/g1_userctrl_23dof_switch eth0 5
```

### 5.3 真实切换并保持30秒

只有短时间测试通过后，才考虑延长时间：

```bash
/home/unitree/bin/g1_userctrl_23dof_switch eth0 30
```

正常输出：

```text
Preflight OK: fsm_id=801, fsm_mode=0, lowstate received, 23 joint targets finite
User control active; holding 23 joints for 5 s
Returned to WALKRUN
```

## 6. 正常物理表现

正常情况下，程序不会让机器人行走。可能观察到：

- 切换瞬间基本不动；
- 关节轻微绷紧；
- 因用户控制刚度与公版运控不同，腿部或手臂出现少量下沉、回弹；
- 吊装机身因关节受力发生轻微晃动；
- 指定时间结束后，公版 WALKRUN重新接管，可能再次出现轻微刚度变化。

## 7. 程序内置保护

程序会在以下情况拒绝切换：

- `fsm_id != 801`；
- `fsm_mode != 0`；
- 两秒内没有收到 `rt/lowstate`；
- 任一有效关节角为 `NaN`、`Inf` 或绝对值大于 `6.5 rad`；
- 参数格式错误；
- 切换前收到 `SIGINT` 或 `SIGTERM`。

控制期间使用独立线程持续以50 Hz发布保持命令，避免同步 API 调用阻塞造成命令断流。

如果 `SwitchToUserCtrl()` 返回错误，程序会把控制状态视为“不确定”，继续发布保持命令并请求恢复到 WALKRUN。

如果切回 WALKRUN失败，程序不会直接退出，而会继续保持命令并重试。长期无法回切时，程序会保持运行，等待人工恢复。

## 8. 安全注意事项

这仍然是真机低层关节控制测试。源码检查、编译成功和 `--dry-run` 通过都不能证明真实切换一定安全。

执行真实切换前必须确认：

- 机器人可靠吊装，双脚完全离地；
- 吊带不会限制腰、髋、肩和手臂运动；
- 机器人四周无人、无障碍物；
- PC1公版运控正常；
- PC2网络和 DDS稳定；
- 没有 `deploy_real`、遥操作或其他低层控制程序同时运行；
- 已准备独立的软件恢复通道，最好同时具备硬件急停。

以下操作不能当作可靠急停：

- 直接关闭 SSH窗口；
- 拔掉 PC2网线；
- `kill -9` 测试程序；
- 停止 DDS发布；
- 重启 PC2。

如果程序提示无法回到 WALKRUN但仍在运行，不要首先杀掉程序。此时它可能仍在持续发布保持命令；突然终止可能造成关节命令断流。

## 9. 常见问题

### 9.1 `No such file or directory`

```text
-bash: /tmp/g1_userctrl_23dof_switch: No such file or directory
```

原因通常是机器人重启后 `/tmp` 被清空。使用持久路径：

```bash
/home/unitree/bin/g1_userctrl_23dof_switch eth0 5
```

### 9.2 FSM状态不匹配

```text
Refusing handoff: expected WALKRUN fsm_id 801, got ...
```

程序没有切换控制权。先确认机器人已经通过公版运控进入 WALKRUN，再重新执行 `--dry-run`。

### 9.3 收不到低层状态

```text
Refusing handoff: no rt/lowstate sample received
```

检查：

- 网卡名是否为 `eth0`；
- PC1与 PC2的控制网络是否连通；
- DDS配置是否正确；
- 是否有网络命名空间或容器隔离。

### 9.4 切回失败

```text
Could not return to WALKRUN after retries
```

程序会继续发布保持命令。应从独立终端或独立设备调用 `SwitchToInternalCtrl(PASSIVE/WALKRUN)`，确认 PC1已接管后再停止当前程序。

## 10. 与 ONNX策略控制的关系

该工具只验证控制权交接和23关节保持，不验证任何强化学习策略。

把 AtomBuildPro、BeyondMimic 或其他 ONNX模型迁移到 User Control 时，还需要额外完成：

- ONNX输入/输出维度检查；
- 29维策略到23个实际关节的映射；
- 缺失6关节的虚拟观测处理；
- 第一帧策略目标的平滑融合；
- 推理超时、姿态异常和网络异常 watchdog；
- 独立恢复程序；
- 23 DoF仿真和吊装验证。

不能仅把 `rt/lowcmd` 改为 `rt/user_lowcmd` 后直接上实机运行策略。
