<script setup lang="ts">
import { onMounted } from "vue";

onMounted(() => {
  void import("./runtime");
});
</script>

<template>
  <div class="shell app-layout">
    <header class="topbar">
      <div class="brand-lockup">
        <span class="brand-chip">G1</span>
        <div>
          <strong>动作预览器</strong>
          <small>29 DoF · 50 Hz · MuJoCo WASM</small>
        </div>
      </div>
      <div class="topbar-meta">
        <span class="frame-readout">
          <span id="frameValue">—</span><small>FRAME</small>
        </span>
        <div class="runtime" data-tone="loading">
          <span class="status-dot"></span><span id="runtimeStatus">LOADING UI</span>
        </div>
      </div>
    </header>

    <main class="workspace">
      <section class="stage-panel">
        <div class="viewport">
          <canvas id="viewportCanvas"></canvas>
          <div id="viewportOverlay" class="viewport-overlay is-loading">
            <div class="loading-spinner" aria-hidden="true"></div>
            <strong id="overlayTitle">正在加载页面</strong>
            <small id="overlayDetail">准备浏览器运行时…</small>
            <div class="loading-bar" aria-hidden="true"><i></i></div>
          </div>
          <div class="viewport-corner">MUJOCO · Z-UP</div>
        </div>
        <div class="transport">
          <button id="playButton" class="primary" disabled>▶ PLAY</button>
          <button id="resetButton" disabled>↺ RESET</button>
          <div class="mode-switch">
            <button data-mode="reference" disabled>REFERENCE</button>
            <button data-mode="policy" class="active" disabled>POLICY</button>
          </div>
          <label class="speed-control">
            SPEED
            <input id="speedInput" type="range" min="0.25" max="2" step="0.25" value="1" />
            <span id="speedValue">1×</span>
          </label>
        </div>
      </section>

      <aside class="side-panel">
        <div id="dropZone" class="drop-zone is-disabled" tabindex="0" role="button" aria-disabled="true">
          <input id="fileInput" type="file" accept=".onnx" hidden />
          <span class="drop-orbit">◎</span>
          <p class="kicker">DROP POLICY</p>
          <h2>拖入 ONNX</h2>
          <p id="dropHint">运行时加载完成后即可导入</p>
          <button type="button" disabled>选择文件</button>
        </div>

        <div class="diagnostics">
          <div class="panel-title"><span>诊断</span><b id="contractBadge">OFFLINE</b></div>
          <div id="diagnosticRows">
            <div class="diag-row"><span>MODEL</span><strong>尚未加载</strong></div>
            <div class="diag-row"><span>OBSERVATION</span><strong>—</strong></div>
            <div class="diag-row"><span>ACTION</span><strong>—</strong></div>
            <div class="diag-row"><span>MUJOCO</span><strong id="mujocoDiag">加载中…</strong></div>
          </div>
        </div>

        <div class="side-status">
          <span class="kicker">状态</span>
          <p id="messageValue" class="message">页面已就绪，正在后台预热物理引擎</p>
        </div>

        <div class="metric-grid">
          <div><small>SIM TIME</small><strong id="timeValue">—</strong><span>sec</span></div>
          <div><small>PELVIS H</small><strong id="heightValue">—</strong><span>m</span></div>
          <div><small>INFERENCE</small><strong id="inferenceValue">—</strong><span>ms</span></div>
          <div><small>MODE</small><strong id="modeValue">—</strong><span>run</span></div>
        </div>

        <div class="log-strip">
          <span class="log-index">LOG</span>
          <p id="logValue">正在加载模块，请稍候…</p>
        </div>
        <p class="side-footnote">
          纯浏览器本地运行 · 仅支持本仓库导出的 G1 motion-policy ONNX
          （obs / time_step / 29 DoF / 内嵌 reference）
        </p>
      </aside>
    </main>
  </div>
</template>