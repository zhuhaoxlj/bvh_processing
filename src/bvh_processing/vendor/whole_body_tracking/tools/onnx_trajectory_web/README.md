# ONNX Trajectory Inspector

本地浏览器工具：拖入 `whole_body_tracking` 导出的 Motion ONNX，读取 `body_pos_w` 和 `body_quat_w`，可切换显示完整参考轨迹，或只显示 pelvis 起点/终点位姿及连接两点的直线。

```bash
cd /home/mark/Documents/Dance/jtq/dance_code/whole_body_tracking
./tools/onnx_trajectory_web/start.sh
```

打开 `http://127.0.0.1:8770`。工具只在本机处理文件，不会启动 MuJoCo、Isaac Sim 或任何机器人控制程序。图中是 ONNX 参考轨迹，不是接触动力学仿真结果。

可通过 `ONNX_TRAJECTORY_PYTHON=/path/to/python` 指定 Python 环境。
