# G1 重定向预览（`retarget_vue`）

浏览器里预览 **BVH 重定向到 Unitree G1 之后的动作轨迹**。

这是纯运动学播放：Three.js 按帧把 JSON 里的关节位姿套到 G1 STL 网格上。**没有重力、没有碰撞、没有物理引擎**。脚穿地或身体悬空都是正常现象——本工具只回答「重定向结果长什么样」，不回答「机器人能不能站住」。

带物理的策略仿真请用旁边的 [`policy_preview_vue`](../policy_preview_vue)。

## 做什么 / 不做什么

- 拖入或选择转换生成的 `*_g1_preview.json`，播放 G1 重定向帧
- 可选再拖入原始 `.bvh`，与 G1 结果并排对比
- **不**做 BVH 解析转换，**不**做重定向计算，**不**跑仿真

网格资源来自项目内的 G1 STL（`/assets/unitree_g1/meshes`）。加载失败时退回点位显示。支持旋转、缩放、播放 / 暂停、回到开头、拖动时间轴。

## 运行

```bash
npm install
npm run dev
```

打开 `http://127.0.0.1:5174/`，把 `*_g1_preview.json` 拖进画布。需要对比时再拖入原始 `.bvh`。

默认端口是 `5174`，与 `policy_preview_vue` 相同，两个项目不要同时 `npm run dev`。

仓库自带示例：

- G1 预览 JSON：`assets/motion_demo/Take_007_049_Skeleton7_g1_preview.json`
- 原始 BVH：`assets/motion_demo/Take_007_049_Skeleton7.bvh`

打包：

```bash
npm run build
npm run preview
```
