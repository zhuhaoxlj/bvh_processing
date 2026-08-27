# 本机 G1 动作跟踪训练 SOP

本文档说明如何在本机 RTX 4090 上启动 G1 动作跟踪训练，将训练和 TensorBoard 放入 `tmux`。

## 1. 当前训练状态

当前动作训练使用以下配置：

- 动作文件：`motions/Take_007_063_Skeleton7_08_g1_retarget_zhrm.npz`
- 任务：`Tracking-Flat-G1-Wo-State-Estimation-v0`
- GPU：RTX 4090，`cuda:0`，24 GB
- 环境数：`18432`（4090 推荐档）
- 最大迭代数：`10000`
- 训练 tmux 会话：`wbt_take063_08_train`
- TensorBoard：tmux 会话 `wbt_tensorboard`，`http://127.0.0.1:6006/`

旧 RTX 5060 Laptop 8 GB 环境实测 `8192` environments 会在 PPO update 阶段 OOM，`7168` 是旧显卡稳定档。
RTX 4090 前端默认使用 `18432`，`16384` 为稳妥档，`12288` 为低压力档。双 RTX 4090 分布式训练实测
`20480` 每卡仍有显存余量，因此前端额外提供 `22528` 高利用率档和 `24576` 激进档。高档位仍可能在 PPO update
阶段 OOM，应先观察两张卡中显存占用更高的一张；环境数越高也不一定吞吐越高，应以 steps/s 和迭代耗时为准。

## 2. 训练前检查

如果训练在环境初始化阶段出现 `Robot body index ... exceeds motion body count ...`，不要降低环境数或修改 NPZ 数组，按 [NPZ 与机器人 Body 顺序不一致排障手册](NPZ_BODY_ORDER_TROUBLESHOOTING.md) 处理。

进入项目：

```bash
cd /home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking
```

确认动作文件存在：

```bash
ls -lh /home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking/motions/Take_007_063_Skeleton7_08_g1_retarget_zhrm.npz
```

确认 GPU 空闲：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits
```

确认没有同项目的旧训练进程：

```bash
pgrep -af 'torch.distributed.run|scripts/rsl_rl/train.py'
```

确认 tmux 会话名没有被占用：

```bash
tmux list-sessions
```

如同名旧会话确实已经无用，可手动关闭：

```bash
tmux kill-session -t wbt_take063_08_train
```

不要在未确认用途前终止其他用户或其他项目的训练进程。

## 3. 启动本机单 GPU 训练

本项目的 Isaac Lab 环境由 `uv` 管理。启动时激活已有 Isaac Lab 虚拟环境，并通过 `uv run --active --no-sync` 执行 Python。

```bash
tmux new-session -d -s wbt_take063_08_train '
  cd /home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking &&
  export PYTHONPATH=/home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking/source/whole_body_tracking &&
  export OMP_NUM_THREADS=8 &&
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True &&
  export VIRTUAL_ENV=/home/mark/Project/01-RL/IsaacLab/env_isaaclab &&
  /home/mark/.local/bin/uv run --active --no-sync \
    python scripts/rsl_rl/train.py \
    --task Tracking-Flat-G1-Wo-State-Estimation-v0 \
    --motion_file /home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking/motions/Take_007_063_Skeleton7_08_g1_retarget_zhrm.npz \
    --num_envs 18432 \
    --max_iterations 10000 \
    --run_name take_007_063_skeleton7_08_local_1gpu_18432 \
    --logger tensorboard \
    --headless \
    --device cuda:0 \
    2>&1 | tee -a /home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking/logs/manual/take_007_063_skeleton7_08_local_1gpu_18432.out
'
```

参数说明：

- `--num_envs 18432`：RTX 4090 推荐起始环境数；首次运行应监控 PPO update 阶段的显存峰值。
- `--headless`：关闭训练渲染，减少显存占用。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：降低 PyTorch 显存碎片风险。
- `tee -a`：将控制台输出同步写入持久日志。

### 显存不足时的调整顺序

如果 `18432` 因其他 GPU 占用或 PPO update 峰值而 OOM：

1. 确认 GPU 上没有其他占用显存的任务；
2. 将环境数降为 `16384`；
3. 仍然 OOM 时继续降为 `12288`；
4. 同步修改 `run_name` 和手工日志文件名，避免不同配置混在一起；
5. 不要开启 `--video`。

## 4. 启动 TensorBoard

Motion Inspector 网页在点击“开始训练”时会先检查 `6006` 端口；如未监听，会自动启动独立的 `wbt_tensorboard` tmux 会话并等待服务就绪，然后才启动训练。手工训练时可使用同样的命令：

```bash
tmux new-session -d -s wbt_tensorboard '
  cd /home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking &&
  export VIRTUAL_ENV=/home/mark/Project/01-RL/IsaacLab/env_isaaclab &&
  /home/mark/.local/bin/uv run --active --no-sync tensorboard \
    --logdir logs/rsl_rl/g1_flat \
    --host 127.0.0.1 \
    --port 6006
'
```

浏览器打开：`http://127.0.0.1:6006/`。检查端口：

```bash
ss -ltnp | rg ':6006\b'
```

## 5. 配置定时停止（可选）

以下命令会在本机时区的下一个 `05:00` 停止训练。如果执行命令时当天 05:00 已经过，则自动安排到次日 05:00。

```bash
tmux new-session -d -s wbt_take063_08_stop_at_0500 '
  target_epoch=$(date -d "today 05:00" +%s)
  current_epoch=$(date +%s)

  if [ "$target_epoch" -le "$current_epoch" ]; then
    target_epoch=$(date -d "tomorrow 05:00" +%s)
  fi

  sleep_seconds=$((target_epoch-current_epoch))
  sleep "$sleep_seconds"

  printf "%s Sending Ctrl-C to wbt_take063_08_train\n" \
    "$(date --iso-8601=seconds)" \
    >> /home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking/logs/manual/take_007_063_stop.log

  tmux send-keys -t wbt_take063_08_train C-c 2>/dev/null || true
  sleep 120

  if tmux has-session -t wbt_take063_08_train 2>/dev/null; then
    printf "%s Force-stopping remaining tmux session\n" \
      "$(date --iso-8601=seconds)" \
      >> /home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking/logs/manual/take_007_063_stop.log
    tmux kill-session -t wbt_take063_08_train
  fi
'
```

该停止机制分为两层：

1. 发送 `Ctrl+C`，优先让 `torchrun` 和 Isaac Sim 正常清理资源；
2. 120 秒后检查训练会话，必要时强制关闭，避免残留训练进程继续占卡。

`tmux` 服务由操作系统维持，因此关闭 Cursor、SSH 或当前终端不会终止训练和定时停止任务。机器重启会清除 tmux 会话；如需跨重启保证执行，应改用 `systemd` timer。

## 6. 启动后验证

确认训练与 TensorBoard 都存在：

```bash
tmux list-sessions
```

正常情况下至少应看到：

```text
wbt_take063_08_train
TensorBoard 监听 127.0.0.1:6006
```

确认训练进程已创建：

```bash
pgrep -af 'torch.distributed.run|scripts/rsl_rl/train.py'
```

查看训练窗口：

```bash
tmux attach -t wbt_take063_08_train
```

从 tmux 中退出但保持训练：按 `Ctrl+B`，松开后按 `D`。

不进入 tmux，直接读取最近输出：

```bash
tmux capture-pane -p -t wbt_take063_08_train -S -150
```

持续查看持久日志：

```bash
tail -f /home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking/logs/manual/take_007_063_skeleton7_08_local_1gpu_7168.out
```

查看 GPU 状态：

```bash
watch -n 2 nvidia-smi
```

启动早期 Isaac Sim 可能长时间进行 CPU 密集的场景创建，GPU 利用率暂时为 0 不一定代表卡死。应同时检查 worker CPU 使用率、日志是否继续增长，以及是否出现 traceback 或 OOM。

快速搜索训练进度和错误：

```bash
rg 'Learning iteration|Total timesteps|CUDA out of memory|OutOfMemory|Traceback|Error executing job' \
  /home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking/logs/manual/take_007_063_skeleton7_08_local_1gpu_7168.out
```

## 7. 验证定时停止任务

确认停止会话存活：

```bash
tmux display-message -p -t wbt_take063_08_stop_at_0500 \
  'command=#{pane_current_command} dead=#{pane_dead}'
```

预期结果中应包含：

```text
command=bash dead=0
```

查看停止任务窗口：

```bash
tmux attach -t wbt_take063_08_stop_at_0500
```

停止动作发生后检查日志：

```bash
tail -n 20 /home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking/logs/manual/take_007_063_stop.log
```

检查是否还有残留进程：

```bash
pgrep -af 'torch.distributed.run|scripts/rsl_rl/train.py'
```

检查 GPU 显存是否已经释放：

```bash
nvidia-smi
```

## 8. 手动提前停止

优先执行正常停止：

```bash
tmux send-keys -t wbt_take063_08_train C-c
```

等待进程清理后确认：

```bash
pgrep -af 'torch.distributed.run|scripts/rsl_rl/train.py'
```

如果训练在合理时间内没有退出，再强制关闭会话：

```bash
tmux kill-session -t wbt_take063_08_train
```

提前手动停止训练后，也应取消尚未触发的定时停止会话：

```bash
tmux kill-session -t wbt_take063_08_stop_at_0500
```

## 9. 日志与 checkpoint

本次控制台日志：

```text
/home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking/logs/manual/take_007_063_skeleton7_08_local_1gpu_7168.out
```

停止任务日志：

```text
/home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking/logs/manual/take_007_063_stop.log
```

RSL-RL 训练目录：

```text
/home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking/logs/rsl_rl/g1_flat/
```

默认配置每 500 次 learning iteration 保存一次 checkpoint。定时停止会保留最近一次已经完整写入磁盘的 checkpoint，但停止发生在两个保存点之间时，中间尚未保存的训练进度不会保留。

查找最近的 checkpoint：

```bash
ls -lt /home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking/logs/rsl_rl/g1_flat/*/model_*.pt
```

## 10. 替换为其他动作

训练其他动作时至少修改以下项目：

1. `--motion_file`；
2. `--run_name`；
3. 训练 tmux 会话名；
4. 停止 tmux 会话名；
5. 手工训练日志文件名；
6. 停止日志文件名；
7. 停止脚本中的目标训练会话名。

建议会话命名格式：

```text
wbt_<动作简称>_train
wbt_<动作简称>_stop_at_<时间>
```

不要复用仍在运行的 tmux 会话名，也不要让两个训练任务写入同一个手工日志文件。
