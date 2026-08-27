# 制作数据








# 训练




# pt 转 onnx

```bash
cd /home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking

export VIRTUAL_ENV=/home/mark/Project/01-RL/IsaacLab/env_isaaclab
export PYTHONPATH="$PWD/source/whole_body_tracking"

MODEL="/home/mark/Downloads/jtq50_2000.pt"
MOTION="/home/mark/Downloads/Take_007_050_Skeleton7_trimmed_recovered_08x.npz"

ls -lh "$MODEL" "$MOTION"

/home/mark/.local/bin/uv run --active --no-sync \
  python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --model_path "$MODEL" \
  --motion_file "$MOTION" \
  --num_envs 1 \
  --device cuda:0 \
  --video \
  --video_length 10 \
  --headless
```

如果崩溃就用这个命令

```bash
/home/mark/.local/bin/uv run --active --no-sync \
  python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --model_path "$MODEL" \
  --motion_file "$MOTION" \
  --num_envs 1 \
  --device cuda:0 \
  --headless
```


# 测试训练好的模型

在仿真里预览
```bash
cd /home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking
/home/mark/Project/01-RL/IsaacLab/env_isaaclab/bin/python \
  scripts/mujoco_sim2sim.py \
  --model /home/mark/Documents/Dance/jtq/models/onnx/jtq107rb_9999.onnx \
  --xml source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml \
  --reset-on-fall \
  --fall-height 0.45
```

播放动作列表
```bash

/home/mark/Project/01-RL/IsaacLab/env_isaaclab/bin/python whole_body_tracking/scripts/mujoco_sim2sim.py --model-dir /home/mark/Documents/Dance/jtq/models/onnx --model-playlist /home/mark/Documents/Dance/jtq/models/onnx/jtq_playlist.txt --xml whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml --record-video --video-path /home/mark/Videos/jtq_playlist_continuous.mp4 --record-all-motions --headless
```


# 启动训练前端页面

```bash
export VIRTUAL_ENV="$HOME/Project/01-RL/IsaacLab/env_isaaclab"
export PYTHONPATH="$PWD/source/whole_body_tracking"

"$VIRTUAL_ENV/bin/python" scripts/npz_preview_web.py --port 8766
```

启动命令会同时确保 TensorBoard 在独立的 `tmux` 会话中运行。默认使用
`http://127.0.0.1:6006/`；如果端口被占用，会依次尝试 `6007-6015`。
TensorBoard 读取 `logs/rsl_rl/g1_flat`，控制台日志写入
`logs/manual/tensorboard_<port>.out`，无需再手动启动。

# 保存 ONNX 仿真视频

无窗口渲染一个完整动作，并编码为 H.264 MP4：

```bash
cd /home/mark/Documents/Dance/jtq/xcct_code/whole_body_tracking

XML="$PWD/source/whole_body_tracking/whole_body_tracking/assets"
XML="$XML/unitree_description/mjcf/g1.xml"

MUJOCO_GL=egl \
  /home/mark/Project/01-RL/IsaacLab/env_isaaclab/bin/python \
  scripts/mujoco_sim2sim.py \
  --model /home/mark/Documents/Dance/jtq/models/onnx/jtq74_4000.onnx \
  --xml "$XML" \
  --headless \
  --record-video \
  --record-one-motion \
  --video-path /home/mark/Videos/jtq74_4000.mp4 \
  --video-width 1280 \
  --video-height 720 \
  --video-fps 50
```

摄像机默认跟随机器人骨盆。可通过 `--camera-distance`、
`--camera-azimuth` 和 `--camera-elevation` 调整机位。若只想录制固定时长，
可将 `--record-one-motion` 换成 `--duration 10`。
