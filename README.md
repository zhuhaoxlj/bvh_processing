# G1 动作预览

本仓库包含两个互不依赖的浏览器预览工具，分别对应动作管线的前后两段：

| | `retarget_vue` | `policy_preview_vue` |
|---|---|---|
| 看什么 | BVH 重定向到 Unitree G1 之后的动作轨迹 | 训练完成后的策略模型仿真 |
| 输入 | `*_g1_preview.json`（可选再叠加原始 `.bvh` 对比） | G1 motion-policy `.onnx` |
| 物理 | **没有**重力、碰撞和物理引擎，只按帧播放轨迹 | **有** MuJoCo 物理引擎：重力、接触、PD 跟踪 |
| 渲染 | Three.js 直接驱动 G1 STL 网格 | MuJoCo WASM 仿真 + Three.js 可视化 |
| 开发端口 | `http://127.0.0.1:5174/` | `http://localhost:5174` |

两个项目都固定使用 `5174` 端口，**不要同时启动**。需要对照时，先停掉其中一个再开另一个。

---

## 该用哪一个

- 只想确认「动捕 / BVH 重定向到 G1 后长什么样」→ 用 [`retarget_vue`](./retarget_vue)。
- 想看「训练好的策略在带重力的仿真里能不能站稳、跟不跟得上参考动作」→ 用 [`policy_preview_vue`](./policy_preview_vue)。

两者都在浏览器本地完成解析与播放，文件不会上传到服务器。

---

## 1. 重定向预览：`retarget_vue`

纯运动学播放器。把已经重定向好的 G1 轨迹按帧套到机器人网格上，**不跑物理**。脚可以穿地、身体可以悬空，这是预期行为。

可选再拖入原始 `.bvh`，和 G1 结果并排对比。前端不负责 BVH 转换或重定向计算。

```bash
cd retarget_vue
npm install
npm run dev
```

打开 `http://127.0.0.1:5174/`，拖入 `*_g1_preview.json`。仓库自带示例：

- G1 预览 JSON：`retarget_vue/assets/motion_demo/Take_007_049_Skeleton7_g1_preview.json`
- 原始 BVH：`retarget_vue/assets/motion_demo/Take_007_049_Skeleton7.bvh`

详细用法见 [retarget_vue/README.md](./retarget_vue/README.md)。

---

## 2. 策略仿真预览：`policy_preview_vue`

带物理的策略预览器。加载训练导出的 ONNX 后，在浏览器里用 MuJoCo WASM 做闭环仿真：策略每 50 Hz 输出 29 维动作，物理以 200 Hz 积分（含重力与接触）。

也可以切到 `REFERENCE` 模式，只看模型内嵌的参考动作。

```bash
cd policy_preview_vue
npm install
npm run dev
```

打开 `http://localhost:5174`，等状态变为 `READY FOR MODEL` 后，把示例策略拖进右侧「拖入 ONNX」区域：

```text
policy_preview_vue/asset/69_43000.onnx
```

推荐 Chrome / Edge。首次打开会加载 WebAssembly 与 G1 模型，需要稍等。

详细用法见 [policy_preview_vue/README.md](./policy_preview_vue/README.md)。

---

## 环境要求

- Node.js 20.19+ 或 22.12+（`policy_preview_vue` 对版本更严格）
- npm
- 支持 WebGL 的现代浏览器；策略预览还需要 WebAssembly
