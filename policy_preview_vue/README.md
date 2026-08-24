# G1 策略仿真预览（`policy_preview_vue`）

浏览器里预览 **训练完成后的 G1 动作策略**。加载兼容的 ONNX 后，用 MuJoCo WASM 做闭环仿真：**有重力、有接触、有物理引擎**。

策略以 50 Hz 输出 29 维动作，物理以 200 Hz 积分。可以对照两种画面：

- `POLICY`：策略控制下的 MuJoCo G1（带重力与接触）
- `REFERENCE`：ONNX 内嵌的参考动作

只想看 BVH 重定向轨迹、不需要物理时，用旁边的 [`retarget_vue`](../retarget_vue)。

技术栈：Vue 3、MuJoCo WASM、ONNX Runtime Web、Three.js。解析和推理都在本机浏览器完成，文件不会上传。

## 环境要求

- Node.js 20.19+ 或 22.12+
- npm
- 推荐最新版 Chrome 或 Edge
- 浏览器需要支持 WebAssembly 和 WebGL

## 启动

```bash
npm install
npm run dev
```

开发服务器固定占用 `5174`：

```text
http://localhost:5174
```

该端口已被占用时会直接报错（例如同时开着 `retarget_vue`）。先关掉占用进程再启动。

首次打开会加载 MuJoCo、ONNX Runtime 和 G1 模型。等右上角变成 `READY FOR MODEL`、右侧「拖入 ONNX」可用后再导入策略。

## 用示例模型跑仿真

仓库自带：

```text
asset/69_43000.onnx
```

1. 打开 `http://localhost:5174`，等 G1 与 MuJoCo 加载完成。
2. 把 `asset/69_43000.onnx` 拖进右侧「拖入 ONNX」，或点「选择文件」。
3. 等解析、协议校验和物理绑定结束。
4. 右上角 `RUNNING`、诊断区 `VALID` 后，点 `PLAY`。

## 预览控制

- `PLAY / PAUSE`：播放或暂停，空格键同样有效。
- `RESET`：停止并回到初始状态。
- `POLICY`：策略控制的物理仿真。
- `REFERENCE`：模型内嵌参考动作。
- `SPEED`：`0.25×`～`2×`。

右侧诊断区显示模型名、观察量维度、动作维度和 MuJoCo 状态；底部显示仿真时间、骨盆高度、单步推理耗时和当前模式。

## 支持的 ONNX

只接受本项目约定的 G1 motion-policy 格式：

- 输入：`obs [1, 154]`、`time_step [1, 1]`
- 输出：`actions [1, 29]`，以及参考关节、速度、位置、姿态、线速度 / 角速度
- 元数据覆盖 29 自由度 G1（刚度、阻尼、默认姿态、动作缩放等）并内嵌参考动作

不兼容时页面会显示 `LOAD ERROR` 或 `REJECTED`，画布和日志里有具体原因。普通 ONNX 不能直接拿来用。

## 构建

```bash
npm run build
npm run preview
```

产物在 `dist/`。预览地址以终端输出为准。

## 常见问题

### 页面一直在加载

首次运行要拉较大的 WASM 和 G1 资源。等一会儿，并看控制台有没有网络、WebAssembly 或 WebGL 错误。优先用最新 Chrome / Edge。

### 拖入后提示模型不兼容

确认文件是 `asset/69_43000.onnx`，或满足上面输入 / 输出 / 元数据约定的导出模型。
