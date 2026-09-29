import * as THREE from 'three';
import { BVHLoader } from '/dev/bvh/static/BVHLoader.js';
import { OrbitControls } from '/dev/bvh/static/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const loader = new BVHLoader();
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const roundTime = (value) => Math.round(value * 20) / 20;

function formatTime(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(seconds, 0) : 0;
  const minutes = Math.floor(safe / 60);
  return `${minutes}:${(safe % 60).toFixed(2).padStart(5, '0')}`;
}

function setStatus(text, kind = '') {
  $('status').textContent = text;
  $('status').className = `status ${kind}`;
}

function log(text, kind = '') {
  const line = document.createElement('span');
  line.className = kind ? `log-${kind}` : '';
  line.textContent = `[${new Date().toLocaleTimeString('zh-CN', { hour12: false })}] ${text}\n`;
  $('activityLog').appendChild(line);
  $('activityLog').scrollTop = $('activityLog').scrollHeight;
}

async function responseError(response) {
  const raw = await response.text();
  try {
    const body = JSON.parse(raw);
    return body.message || body.detail?.[0]?.msg || body.detail || raw;
  } catch {
    return raw || `HTTP ${response.status}`;
  }
}

function createViewer(canvas, color) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0c1014);
  const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 100000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = true;

  let root = null;
  let helper = null;
  let grid = null;
  let axes = null;
  let mixer = null;
  let action = null;
  let duration = 0;
  let frames = 0;
  const homePosition = new THREE.Vector3(2, 1, 2);
  const homeTarget = new THREE.Vector3();

  /**
   * 把动作定位到绝对时间。动作统一用 LoopOnce + clampWhenFinished：
   * 默认的 LoopRepeat 在 time 恰好等于 clip.duration 时会回卷到第 0 帧，
   * 表现为拖到/播到最后一帧时骨架“瞬移”回起始姿态；命中末尾后动作会停在
   * 结束状态，所以每次定位前先 reset 才能继续往回拖。
   */
  function seek(targetAction, targetMixer, seconds) {
    targetAction.reset();
    targetMixer.setTime(seconds);
  }

  function dispose(object) {
    if (!object) return;
    object.traverse((node) => {
      node.geometry?.dispose();
      if (node.material) {
        for (const material of Array.isArray(node.material) ? node.material : [node.material]) material.dispose();
      }
    });
  }

  function clear() {
    for (const object of [root, helper, grid, axes]) {
      if (object) scene.remove(object);
      dispose(object);
    }
    root = helper = grid = axes = mixer = action = null;
    duration = 0;
    frames = 0;
  }

  function load(text) {
    clear();
    const parsed = loader.parse(text);
    root = parsed.skeleton.bones[0];
    if (!root) throw new Error('BVH 中没有骨架');
    helper = new THREE.SkeletonHelper(root);
    helper.material.color.setHex(color);
    scene.add(root, helper);
    mixer = new THREE.AnimationMixer(root);
    action = mixer.clipAction(parsed.clip);
    action.setLoop(THREE.LoopOnce, 1);
    action.clampWhenFinished = true;
    action.play();
    duration = parsed.clip.duration;
    frames = parsed.clip.tracks[0]?.times.length || 1;

    const bones = [];
    root.traverse((node) => { if (node.isBone) bones.push(node); });
    const box = new THREE.Box3();
    const point = new THREE.Vector3();
    const probe = new THREE.AnimationMixer(root);
    const probeAction = probe.clipAction(parsed.clip);
    probeAction.setLoop(THREE.LoopOnce, 1);
    probeAction.clampWhenFinished = true;
    probeAction.play();
    for (let index = 0; index <= 20; index += 1) {
      seek(probeAction, probe, duration * index / 20);
      root.updateMatrixWorld(true);
      for (const bone of bones) box.expandByPoint(point.setFromMatrixPosition(bone.matrixWorld));
    }
    probe.stopAllAction();
    if (box.isEmpty()) box.setFromCenterAndSize(new THREE.Vector3(), new THREE.Vector3(1, 1, 1));
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z, 1) * .5;

    grid = new THREE.GridHelper(radius * 3, 12, 0x35404c, 0x202832);
    grid.position.y = box.min.y;
    axes = new THREE.AxesHelper(radius * .65);
    axes.position.y = box.min.y;
    scene.add(grid, axes);

    homePosition.set(center.x + radius * 2.1, center.y + radius * .45, center.z + radius * 2.1);
    homeTarget.copy(center);
    camera.near = Math.max(radius / 200, .001);
    camera.far = radius * 300;
    camera.updateProjectionMatrix();
    resetView();
    return { duration, frames, bones: bones.length };
  }

  function resetView() {
    camera.position.copy(homePosition);
    controls.target.copy(homeTarget);
    controls.update();
  }

  function render(progress) {
    const width = canvas.clientWidth || 1;
    const height = canvas.clientHeight || 1;
    const actual = renderer.getSize(new THREE.Vector2());
    if (actual.x !== width || actual.y !== height) {
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }
    if (mixer && action) seek(action, mixer, Math.min(Math.max(progress, 0), 1) * duration);
    controls.update();
    renderer.render(scene, camera);
  }

  return {
    load,
    clear,
    resetView,
    render,
    get duration() { return duration; },
    get frames() { return frames; },
    get loaded() { return root !== null; },
  };
}

const sourceViewer = createViewer($('sourceCanvas'), 0x4d9fff);
const resultViewer = createViewer($('resultCanvas'), 0x42d392);
let previewProgress = 0;
let playing = true;
let lastFrameAt = performance.now();

function previewDuration() {
  return Math.max(sourceViewer.duration, resultViewer.duration, 0);
}

function updateTimelinePlayhead(follow = false) {
  const playhead = $('timelinePlayhead');
  const selected = clips.find((clip) => clip.id === selectedId);
  playhead.hidden = !clips.length || (!resultViewer.loaded && (!selected || !sourceViewer.loaded));
  if (playhead.hidden) return;

  const time = resultViewer.loaded
    ? previewProgress * previewDuration()
    : selected.start + previewProgress * sourceViewer.duration;
  playhead.style.left = `calc(var(--track-label) + ${time * pixelsPerSecond}px)`;

  if (follow) {
    const scroll = $('timelineScroll');
    const x = $('track').offsetLeft + time * pixelsPerSecond;
    if (x < scroll.scrollLeft + 20 || x > scroll.scrollLeft + scroll.clientWidth - 20) {
      scroll.scrollLeft = Math.max(0, x - scroll.clientWidth / 2);
    }
  }
}

function updateTransport(follow = false) {
  $('previewProgress').value = String(Math.round(previewProgress * 1000));
  $('previewTime').textContent = `${formatTime(previewProgress * previewDuration())} / ${formatTime(previewDuration())}`;
  $('playButton').textContent = playing ? 'Ⅱ' : '▶';
  updateTimelinePlayhead(follow);
}

function animate(now) {
  requestAnimationFrame(animate);
  const delta = Math.min((now - lastFrameAt) / 1000, .1);
  lastFrameAt = now;
  const duration = previewDuration();
  if (playing && duration > 0) previewProgress = (previewProgress + delta / duration) % 1;
  sourceViewer.render(previewProgress);
  resultViewer.render(previewProgress);
  updateTransport(playing);
}
requestAnimationFrame(animate);

$('playButton').addEventListener('click', () => { playing = !playing; updateTransport(); });
$('previewProgress').addEventListener('input', (event) => {
  previewProgress = Number(event.target.value) / 1000;
  playing = false;
  updateTransport(true);
});
$('resetViewButton').addEventListener('click', () => {
  sourceViewer.resetView();
  resultViewer.resetView();
});

let clips = [];
let selectedId = null;
let pixelsPerSecond = 90;
let busy = false;
let resultUrl = null;

function timelineEnd() {
  return clips.reduce((end, clip) => Math.max(end, clip.start + clip.duration), 0);
}

function gapAfter(index) {
  if (index >= clips.length - 1) return 0;
  return roundTime(Math.max(0, clips[index + 1].start - clips[index].start - clips[index].duration));
}

function timelineError() {
  for (let index = 0; index < clips.length - 1; index += 1) {
    const gap = gapAfter(index);
    if (gap > 0 && gap < .05) return `片段 ${index + 1} 后的间隔不能小于 0.05 秒`;
    if (gap > 7.8) return `片段 ${index + 1} 后的间隔不能超过 7.8 秒`;
  }
  return '';
}

function payload(uploaded = false) {
  const origin = window.location.origin;
  return {
    actionId: `dev-merge-${Date.now()}`,
    timelineOffsetSec: roundTime(clips[0]?.start || 0),
    segments: clips.map((clip, index) => ({
      segmentId: clip.id,
      actionId: index + 1,
      actionUrl: clip.upload ? origin + clip.upload.sourcePath : (uploaded ? '' : `[上传后生成] ${clip.file.name}`),
      sourceInSec: 0,
      sourceOutSec: null,
      outputDurationSec: roundTime(clip.duration),
      gapAfterSec: gapAfter(index),
    })),
    callbackUrl: `${origin}/api/v1/dev/bvh/callback/[运行时生成]`,
  };
}

function renderPayload() {
  $('payloadPreview').textContent = JSON.stringify(payload(), null, 2);
}

function renderInspector() {
  const clip = clips.find((item) => item.id === selectedId);
  $('removeButton').disabled = !clip || busy;
  const meta = $('segmentMeta');
  meta.textContent = '';
  const entries = clip ? [
    ['文件', clip.file.name],
    ['大小', `${(clip.file.size / 1024).toFixed(1)} KB`],
    ['开始', `${clip.start.toFixed(2)} s`],
    ['时长', `${clip.duration.toFixed(2)} s`],
    ['帧数', clip.frames],
    ['关节', clip.bones],
  ] : [['状态', '选择时间轴中的片段']];
  for (const [key, value] of entries) {
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = key;
    dd.textContent = String(value);
    meta.append(dt, dd);
  }
}

function selectClip(id) {
  selectedId = id;
  const clip = clips.find((item) => item.id === id);
  if (clip) {
    try {
      sourceViewer.load(clip.text);
      $('sourceInfo').textContent = `${clip.file.name} · ${clip.duration.toFixed(2)}s · ${clip.frames} 帧`;
      previewProgress = 0;
    } catch (error) {
      setStatus(`预览失败：${error.message}`, 'error');
    }
  }
  renderTimeline();
  updateTransport(true);
  renderInspector();
}

function renderRuler(width, seconds) {
  const ruler = $('ruler');
  ruler.textContent = '';
  const step = pixelsPerSecond >= 100 ? .5 : 1;
  for (let time = 0; time <= seconds + step; time += step) {
    const tick = document.createElement('span');
    const major = Math.abs(time - Math.round(time)) < .001;
    tick.className = `tick ${major ? 'major' : ''}`;
    tick.style.left = `${time * pixelsPerSecond}px`;
    tick.textContent = major ? `${Math.round(time)}s` : '';
    ruler.appendChild(tick);
  }
  ruler.style.width = `${width}px`;
}

function beginDrag(event, id) {
  if (busy || event.button !== 0 || event.target.closest('.clip-remove')) return;
  event.preventDefault();
  selectClip(id);
  const index = clips.findIndex((clip) => clip.id === id);
  const pointerStart = event.clientX;
  const starts = clips.map((clip) => clip.start);
  const previousEnd = index === 0 ? 0 : clips[index - 1].start + clips[index - 1].duration;
  const minDelta = previousEnd - starts[index];
  const target = event.currentTarget;
  target.classList.add('dragging');

  function move(moveEvent) {
    let delta = roundTime((moveEvent.clientX - pointerStart) / pixelsPerSecond);
    delta = Math.max(delta, minDelta);
    for (let cursor = index; cursor < clips.length; cursor += 1) clips[cursor].start = roundTime(starts[cursor] + delta);
    renderTimeline();
    renderInspector();
    renderPayload();
  }

  function end() {
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', end);
    window.removeEventListener('pointercancel', end);
  }
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', end);
  window.addEventListener('pointercancel', end);
}

function removeClip(id) {
  const index = clips.findIndex((clip) => clip.id === id);
  if (index < 0) return;
  const [removed] = clips.splice(index, 1);
  if (selectedId === id) {
    selectedId = clips[Math.min(index, clips.length - 1)]?.id || null;
    if (selectedId) selectClip(selectedId);
    else {
      sourceViewer.clear();
      $('sourceInfo').textContent = '未选择';
    }
  }
  log(`移除片段 ${removed.file.name}`);
  renderAll();
}

function renderTimeline() {
  const track = $('track');
  track.querySelectorAll('.clip, .gap').forEach((node) => node.remove());
  $('dropEmpty').hidden = clips.length > 0;
  const seconds = Math.max(12, Math.ceil(timelineEnd() + 2));
  const width = seconds * pixelsPerSecond;
  $('timelineStage').style.width = `calc(var(--track-label) + ${width}px)`;
  track.style.width = `${width}px`;
  renderRuler(width, seconds);

  clips.forEach((clip, index) => {
    if (index > 0) {
      const previous = clips[index - 1];
      const gapValue = gapAfter(index - 1);
      if (gapValue > 0) {
        const gap = document.createElement('div');
        gap.className = `gap ${gapValue > 7.8 ? 'invalid' : ''}`;
        gap.style.left = `${(previous.start + previous.duration) * pixelsPerSecond}px`;
        gap.style.width = `${gapValue * pixelsPerSecond}px`;
        gap.textContent = `${gapValue.toFixed(2)}s`;
        track.appendChild(gap);
      }
    }

    const node = document.createElement('div');
    node.className = `clip ${clip.id === selectedId ? 'selected' : ''}`;
    node.style.left = `${clip.start * pixelsPerSecond}px`;
    node.style.width = `${Math.max(clip.duration * pixelsPerSecond, 30)}px`;
    node.title = `${clip.file.name}\n开始 ${clip.start.toFixed(2)}s · 时长 ${clip.duration.toFixed(2)}s`;
    node.innerHTML = `<span class="clip-index">${String(index + 1).padStart(2, '0')}</span><span class="clip-copy"><span class="clip-name"></span><span class="clip-duration"></span></span><button class="clip-remove" type="button" title="删除片段" aria-label="删除片段">×</button>`;
    node.querySelector('.clip-name').textContent = clip.file.name;
    node.querySelector('.clip-duration').textContent = `${clip.duration.toFixed(2)}s`;
    node.addEventListener('click', () => selectClip(clip.id));
    node.addEventListener('pointerdown', (event) => beginDrag(event, clip.id));
    node.querySelector('.clip-remove').addEventListener('click', (event) => {
      event.stopPropagation();
      removeClip(clip.id);
    });
    track.appendChild(node);
  });

  const totalGap = clips.slice(0, -1).reduce((sum, _, index) => sum + gapAfter(index), 0);
  $('timelineSummary').textContent = `${clips.length} 个片段 · ${timelineEnd().toFixed(2)}s · 过渡 ${totalGap.toFixed(2)}s`;
  const error = timelineError();
  $('mergeButton').disabled = busy || clips.length === 0 || Boolean(error);
  if (error && !busy) setStatus(error, 'error');
  updateTimelinePlayhead();
}

function renderAll() {
  renderTimeline();
  renderInspector();
  renderPayload();
}

async function parseFile(file) {
  if (!file.name.toLowerCase().endsWith('.bvh')) throw new Error(`${file.name} 不是 BVH 文件`);
  const text = await file.text();
  const parsed = loader.parse(text);
  const duration = parsed.clip.duration;
  if (!(duration > 0)) throw new Error(`${file.name} 没有可播放的动作帧`);
  let bones = 0;
  parsed.skeleton.bones[0]?.traverse((node) => { if (node.isBone) bones += 1; });
  return {
    id: `segment-${crypto.randomUUID ? crypto.randomUUID() : Date.now()}-${Math.random().toString(16).slice(2)}`,
    file,
    text,
    duration,
    frames: parsed.clip.tracks[0]?.times.length || 1,
    bones,
    start: roundTime(timelineEnd()),
    upload: null,
  };
}

async function addFiles(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  setStatus(`解析 ${files.length} 个 BVH…`, 'busy');
  for (const file of files) {
    try {
      const clip = await parseFile(file);
      clip.start = roundTime(timelineEnd());
      clips.push(clip);
      log(`添加 ${file.name}（${clip.duration.toFixed(2)}s，${clip.frames} 帧）`, 'info');
      if (!selectedId) selectedId = clip.id;
    } catch (error) {
      log(error.message, 'error');
    }
  }
  if (selectedId) selectClip(selectedId);
  setStatus(clips.length ? '拖动片段调整间隔' : '没有可用片段');
  renderAll();
}

$('fileInput').addEventListener('change', (event) => {
  addFiles(event.target.files).finally(() => { event.target.value = ''; });
});
for (const type of ['dragenter', 'dragover']) {
  $('track').addEventListener(type, (event) => { event.preventDefault(); $('track').classList.add('drag-over'); });
}
for (const type of ['dragleave', 'drop']) $('track').addEventListener(type, () => $('track').classList.remove('drag-over'));
$('track').addEventListener('drop', (event) => { event.preventDefault(); addFiles(event.dataTransfer.files); });
$('zoomInput').addEventListener('input', (event) => { pixelsPerSecond = Number(event.target.value); renderTimeline(); updateTimelinePlayhead(true); });
window.addEventListener('resize', () => updateTimelinePlayhead(true));
$('compactButton').addEventListener('click', () => {
  let cursor = 0;
  for (const clip of clips) { clip.start = roundTime(cursor); cursor += clip.duration; }
  renderAll();
  setStatus('已移除全部片段间隔');
});
$('removeButton').addEventListener('click', () => { if (selectedId) removeClip(selectedId); });
$('clearLogButton').addEventListener('click', () => { $('activityLog').textContent = ''; });
$('copyButton').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText($('payloadPreview').textContent);
    $('copyButton').textContent = '已复制';
  } catch {
    $('copyButton').textContent = '复制失败';
  }
  setTimeout(() => { $('copyButton').textContent = '复制 JSON'; }, 1200);
});
$('downloadButton').addEventListener('click', () => {
  if (!resultUrl) return;
  $('downloadAnchor').href = resultUrl;
  $('downloadAnchor').download = '';
  $('downloadAnchor').click();
});

async function uploadClip(clip) {
  if (clip.upload) return clip.upload;
  const body = new FormData();
  body.append('file', clip.file, clip.file.name);
  const response = await fetch('/api/v1/dev/bvh/uploads', { method: 'POST', body });
  if (!response.ok) throw new Error(`${clip.file.name} 上传失败：${await responseError(response)}`);
  clip.upload = await response.json();
  log(`上传完成 ${clip.file.name}`, 'info');
  return clip.upload;
}

async function pollTask(taskId, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const response = await fetch(`/api/v1/dev/bvh/tasks/${taskId}`, { cache: 'no-store' });
    if (response.ok) {
      const task = await response.json();
      if (task.status !== 'pending') return task;
    }
    await sleep(750);
  }
  throw new Error('等待合并回调超时');
}

async function runMerge() {
  if (busy || !clips.length) return;
  const validation = timelineError();
  if (validation) throw new Error(validation);
  busy = true;
  resultUrl = null;
  $('downloadButton').disabled = true;
  renderTimeline();
  setStatus(`上传 0/${clips.length}`, 'busy');

  for (let index = 0; index < clips.length; index += 1) {
    await uploadClip(clips[index]);
    setStatus(`上传 ${index + 1}/${clips.length}`, 'busy');
  }

  const devTaskId = `merge-${crypto.randomUUID ? crypto.randomUUID() : Date.now()}`;
  const request = payload(true);
  request.callbackUrl = `${window.location.origin}/api/v1/dev/bvh/callback/${devTaskId}`;
  $('payloadPreview').textContent = JSON.stringify(request, null, 2);
  setStatus('MDM 合并处理中…', 'busy');
  log(`POST /api/v1/bvh/merge，${clips.length} 个片段`, 'info');
  const response = await fetch('/api/v1/bvh/merge', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error(`merge 接口错误：${await responseError(response)}`);
  const accepted = await response.json();
  log(`接口已接收 taskId=${accepted.taskId}`, 'info');

  const task = await pollTask(devTaskId, 10 * 60 * 1000);
  if (task.status !== 'succeeded') throw new Error(task.message || '合并失败');
  const result = await fetch(task.result.url, { cache: 'no-store' });
  if (!result.ok) throw new Error(`结果下载失败：${await responseError(result)}`);
  const info = resultViewer.load(await result.text());
  resultUrl = task.result.url;
  $('resultInfo').textContent = `${task.result.filename} · ${info.duration.toFixed(2)}s · ${info.frames} 帧`;
  $('downloadButton').disabled = false;
  previewProgress = 0;
  playing = true;
  setStatus('合并成功', 'ok');
  log(`合并成功 ${task.result.filename}（${task.result.size} B）`, 'ok');
}

$('mergeButton').addEventListener('click', () => {
  runMerge().catch((error) => {
    setStatus(error.message, 'error');
    log(error.message, 'error');
  }).finally(() => {
    busy = false;
    renderTimeline();
  });
});

renderAll();
log('拖入多个 BVH，拖动片段制造间隔，然后调用 merge 接口。');
