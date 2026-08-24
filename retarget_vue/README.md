# G1 Motion Preview (Vue)

独立的前端预览页，只做两件事：

- 拖入或选择转换生成的 `*_g1_preview.json`，预览 G1 重定向帧；
- 可选拖入原始 `.bvh`，与 G1 JSON 同步对比。

前端只负责读取和预览文件，不负责 BVH 转换或重定向计算。预览使用项目内的 G1 STL 网格作为显示资源（加载失败时退回点位），支持视角旋转、缩放、播放/暂停、回到开头和时间轴拖动。运行时用户只需提供 G1 预览 JSON；原始 BVH 用于可选对比，不依赖 Python 后端。

## 运行

```bash
npm install
npm run dev
```

打开 `http://127.0.0.1:5174/`，将 `*_g1_preview.json` 拖入画布；如需对比，再拖入原始 `.bvh`。JSON 里的网格地址指向项目内的 `/assets/unitree_g1/meshes`，这些 STL 文件由前端自动加载，不需要用户单独选择。

打包静态资源：

```bash
npm run build
npm run preview
```
