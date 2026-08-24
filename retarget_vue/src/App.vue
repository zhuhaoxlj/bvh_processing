<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import { G1Preview, parseBvh } from "./g1-preview.js";

const canvas = ref(null);
const jsonFile = ref(null);
const bvhFile = ref(null);
const motion = ref(null);
const sourceBvh = ref(null);
const loading = ref(false);
const error = ref("");
const playing = ref(false);
const frame = ref(0);
const viewer = ref(null);

function ensureViewer() {
  if (viewer.value) return viewer.value;
  if (!canvas.value) throw new Error("预览画布尚未初始化，请刷新页面后重试。");
  viewer.value = new G1Preview(canvas.value, {
    onFrame: (next) => { frame.value = next; },
    onPlaying: (next) => { playing.value = next; },
  });
  return viewer.value;
}

const frameCount = computed(() => motion.value?.frames?.length || 0);
const currentTime = computed(() => {
  const fps = Number(motion.value?.fps || 30);
  return `${(frame.value / fps).toFixed(2)}s / ${(Math.max(0, frameCount.value - 1) / fps).toFixed(2)}s`;
});

async function applyMotion(nextMotion, nextBvh = null) {
  const preview = ensureViewer();
  motion.value = nextMotion;
  sourceBvh.value = nextBvh ? bvhFile.value?.name || "source.bvh" : null;
  await preview.setMotion(nextMotion, nextBvh);
  frame.value = 0;
  playing.value = true;
  preview.setPlaying(true);
}

async function loadJson(file = jsonFile.value) {
  if (!file) {
    error.value = "请拖入或选择预览 JSON 文件。";
    return;
  }
  loading.value = true;
  error.value = "";
  try {
    await nextTick();
    const data = JSON.parse(await file.text());
    if (!data?.frames?.length || !data?.mesh_geoms?.length) throw new Error("JSON 不是有效的 G1 预览文件。请使用转换生成的 *_g1_preview.json。");
    const bvh = bvhFile.value ? parseBvh(await bvhFile.value.text()) : null;
    await applyMotion(data, bvh);
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
    playing.value = false;
  } finally {
    loading.value = false;
  }
}

async function loadBvh(file) {
  try {
    bvhFile.value = file;
    if (jsonFile.value) await applyMotion(JSON.parse(await jsonFile.value.text()), parseBvh(await file.text()));
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  }
}

function chooseFile(event, expected = null) {
  const file = event.target.files?.[0];
  if (!file) return;
  const isBvh = file.name.toLowerCase().endsWith(".bvh");
  if (expected && ((expected === "bvh") !== isBvh)) {
    error.value = expected === "bvh" ? "这里请选择 .bvh 文件。" : "这里请选择 *_g1_preview.json 文件。";
    return;
  }
  if (isBvh) loadBvh(file);
  else { jsonFile.value = file; loadJson(file); }
}

function handleDrop(event, expected = null) {
  event.preventDefault();
  const file = event.dataTransfer.files?.[0];
  if (!file) return;
  const isBvh = file.name.toLowerCase().endsWith(".bvh");
  if (expected && ((expected === "bvh") !== isBvh)) {
    error.value = expected === "bvh" ? "这里请拖入 .bvh 文件。" : "这里请拖入 *_g1_preview.json 文件。";
  } else if (isBvh) loadBvh(file);
  else if (file.name.toLowerCase().endsWith(".json")) { jsonFile.value = file; loadJson(file); }
  else error.value = "请拖入 *_g1_preview.json 或 .bvh 文件。";
}

function setPlaying(value) {
  playing.value = value;
  viewer.value?.setPlaying(value);
}

function seek(event) {
  const next = Number(event.target.value);
  frame.value = next;
  viewer.value?.setFrame(next);
  setPlaying(false);
}

onMounted(() => {
  try {
    ensureViewer();
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  }
});

onBeforeUnmount(() => viewer.value?.dispose());
</script>

<template>
  <main class="shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">UNITREE G1 / RETARGET</p>
        <h1>G1 动作预览</h1>
        <p class="subline">JSON 重定向结果与可选 BVH 原始动作对比</p>
      </div>
      <div class="load-form">
        <input id="json-file" class="file-input" type="file" accept=".json" @change="chooseFile($event, 'json')" />
        <label for="json-file" class="file-button">选择 JSON</label>
        <input id="bvh-file" class="file-input" type="file" accept=".bvh" @change="chooseFile($event, 'bvh')" />
        <label for="bvh-file" class="file-button secondary-file">选择 BVH</label>
      </div>
    </header>

    <section class="workspace">
      <div class="viewer-panel" @dragover.prevent @drop="handleDrop">
        <canvas ref="canvas" aria-label="G1 motion preview"></canvas>
        <div v-if="!jsonFile" class="drop-hint">拖入 G1 预览 JSON</div>
        <div v-if="error" class="error">{{ error }}</div>
        <div class="legend">
          <span class="g1-dot"></span> G1 JSON
          <span v-if="sourceBvh" class="bvh-dot"></span> {{ sourceBvh }}
        </div>
      </div>

      <aside class="controls">
        <div class="drop-zones">
          <div class="drop-zone" :class="{ loaded: jsonFile }" @dragover.prevent @drop="handleDrop($event, 'json')">
            <input id="json-drop-file" class="file-input" type="file" accept=".json" @change="chooseFile($event, 'json')" />
            <label for="json-drop-file"><strong>{{ jsonFile ? "JSON 已加载" : "G1 预览 JSON" }}</strong><span>{{ jsonFile?.name || "拖入或点击选择" }}</span></label>
          </div>
          <div class="drop-zone" :class="{ loaded: bvhFile }" @dragover.prevent @drop="handleDrop($event, 'bvh')">
            <input id="bvh-drop-file" class="file-input" type="file" accept=".bvh" @change="chooseFile($event, 'bvh')" />
            <label for="bvh-drop-file"><strong>{{ bvhFile ? "BVH 已加载" : "原始 BVH 对比" }}</strong><span>{{ bvhFile?.name || "可选，拖入或点击选择" }}</span></label>
          </div>
        </div>
        <div class="identity">
          <span class="status">{{ motion ? "READY" : "IDLE" }}</span>
          <h2>{{ motion?.name || "等待 G1 JSON" }}</h2>
          <p>{{ motion ? `${motion.robot} · ${motion.fps || 30} FPS` : "拖入转换生成的预览 JSON" }}</p>
        </div>
        <div class="transport">
          <button class="play" :disabled="!motion" @click="setPlaying(!playing)">{{ playing ? "暂停" : "播放" }}</button>
          <button :disabled="!motion" @click="viewer?.setFrame(0); setPlaying(false)">回到开头</button>
        </div>
        <input class="scrubber" type="range" min="0" :max="Math.max(0, frameCount - 1)" :value="frame" :disabled="!motion" @input="seek" />
        <div class="time"><span>FRAME {{ frameCount ? frame + 1 : 0 }} / {{ frameCount }}</span><strong>{{ currentTime }}</strong></div>
        <dl v-if="motion" class="stats">
          <div><dt>身体节点</dt><dd>{{ motion.body_names?.length || 0 }}</dd></div>
          <div><dt>帧数</dt><dd>{{ frameCount }}</dd></div>
          <div><dt>预览模式</dt><dd>{{ motion.mesh_geoms?.length ? "STL 网格" : "点位" }}</dd></div>
          <div><dt>对比 BVH</dt><dd>{{ sourceBvh ? "已加载" : "无" }}</dd></div>
        </dl>
      </aside>
    </section>
  </main>
</template>
