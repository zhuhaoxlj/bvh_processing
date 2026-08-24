# G1 Motion Preview (Vue)

独立的前端预览页，只做两件事：

- 拖入或选择转换生成的 `*_g1_preview.json`，直接读取 G1 重定向帧；
- 可再拖入原始 `.bvh`，与 G1 JSON 同步对比。

预览使用项目内的 G1 STL 网格（加载失败时退回点位），支持 OrbitControls、播放/暂停、回到开头和时间轴拖动。前端运行时只读 JSON 和 BVH，不依赖 Python 后端。

## 运行

```bash
npm install
npm run dev
```

打开 `http://127.0.0.1:5174/`，将 `*_g1_preview.json` 拖入画布；如需对比，再拖入原始 `.bvh`。JSON 里的网格地址指向项目内的 `/assets/unitree_g1/meshes`，因此运行时只需要 Vite 静态文件服务。

打包静态资源：

```bash
npm run build
npm run preview
```
