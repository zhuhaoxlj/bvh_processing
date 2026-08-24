import type { G1Controller } from "./simulation/controller";
import type { G1Model } from "./simulation/mujoco";
import type { G1Scene } from "./rendering/scene";
import type { PolicyLoadResult, PreviewMode, SimulationSnapshot } from "./types";

const $ = <T extends Element>(selector: string) => document.querySelector<T>(selector)!;
const runtimeStatus = $("#runtimeStatus") as HTMLElement;
const overlay = $("#viewportOverlay") as HTMLElement;
const playButton = $("#playButton") as HTMLButtonElement;
const resetButton = $("#resetButton") as HTMLButtonElement;
const fileInput = $("#fileInput") as HTMLInputElement;
const dropZone = $("#dropZone") as HTMLElement;
const dropButton = dropZone.querySelector("button") as HTMLButtonElement;
const dropHint = $("#dropHint") as HTMLElement;
const logValue = $("#logValue") as HTMLElement;
const mujocoDiag = $("#mujocoDiag") as HTMLElement;
const canvas = $("#viewportCanvas") as HTMLCanvasElement;

const CONTROL_DT = 0.02;
const MAX_CATCH_UP_STEPS = 5;

let scene: G1Scene | null = null;
let loaded: PolicyLoadResult | null = null;
let controller: G1Controller | null = null;
let sim: G1Model | null = null;
let sceneBound = false;
let lastTime = 0;
let accumulator = 0;
let speed = 1;
let busy = false;
let tickInFlight = false;

let loadPolicy!: typeof import("./policy/loader").loadPolicy;
let preloadOrtRuntime!: typeof import("./policy/loader").preloadOrtRuntime;
let createG1Model!: typeof import("./simulation/mujoco").createG1Model;
let disposeG1!: typeof import("./simulation/mujoco").disposeG1;
let G1_POLICY_JOINTS!: typeof import("./simulation/mujoco").G1_POLICY_JOINTS;
let isG1RuntimeReady!: typeof import("./simulation/mujoco").isG1RuntimeReady;
let preloadG1Runtime!: typeof import("./simulation/mujoco").preloadG1Runtime;
let rebindPolicyJoints!: typeof import("./simulation/mujoco").rebindPolicyJoints;
let startG1Prefetch!: typeof import("./simulation/mujoco").startG1Prefetch;
let G1ControllerCtor!: typeof import("./simulation/controller").G1Controller;

// Kick WASM+MJB downloads immediately after shell HTML is in the DOM — do not
// wait for Three.js / ORT chunks to finish parsing.
const earlyPrefetch = import("./simulation/mujoco")
  .then((mod) => {
    startG1Prefetch = mod.startG1Prefetch;
    return mod.startG1Prefetch();
  })
  .catch(() => null);

function setStatus(message: string, tone: "idle" | "ready" | "error" | "loading" = "idle"): void {
  runtimeStatus.textContent = message;
  runtimeStatus.parentElement!.dataset.tone = tone;
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character] ?? character);
}

function setLoading(title: string, detail: string, status = title.toUpperCase()): void {
  overlay.classList.remove("hidden");
  overlay.classList.add("is-loading");
  overlay.innerHTML = `
    <div class="loading-spinner" aria-hidden="true"></div>
    <strong id="overlayTitle">${escapeHtml(title)}</strong>
    <small id="overlayDetail">${escapeHtml(detail)}</small>
    <div class="loading-bar" aria-hidden="true"><i></i></div>
  `;
  setStatus(status, "loading");
  logValue.textContent = detail;
  mujocoDiag.textContent = title;
}

function setOverlayReady(title: string, detail: string): void {
  overlay.classList.remove("hidden", "is-loading");
  overlay.innerHTML = `<span class="crosshair">＋</span><strong>${escapeHtml(title)}</strong><small>${escapeHtml(detail)}</small>`;
}

function showError(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  setStatus("LOAD ERROR", "error");
  logValue.textContent = message;
  overlay.classList.remove("hidden", "is-loading");
  overlay.innerHTML = `<span class="crosshair error-mark">×</span><strong>模型不兼容</strong><small>${escapeHtml(message)}</small>`;
  $("#contractBadge").textContent = "REJECTED";
  $("#contractBadge").className = "bad";
}

function setDropEnabled(enabled: boolean): void {
  dropZone.classList.toggle("is-disabled", !enabled);
  dropZone.setAttribute("aria-disabled", enabled ? "false" : "true");
  dropButton.disabled = !enabled;
  dropHint.textContent = enabled
    ? "或点击选择一个本仓库导出的策略模型"
    : "运行时加载完成后即可导入策略";
}

function bindScene(nextSim: G1Model): void {
  if (!scene) return;
  if (sceneBound && sim === nextSim) {
    nextSim.mujoco.mj_resetData(nextSim.model, nextSim.data);
    return;
  }
  scene.load(nextSim);
  sceneBound = true;
}

async function warmUpRuntime(): Promise<void> {
  setLoading("正在预热 MuJoCo", "预取 WASM 与预编译 G1 模型…", "WARMING UP");
  preloadOrtRuntime();
  try {
    const t0 = performance.now();
    // Ensure early prefetch has been started (idempotent).
    await earlyPrefetch;
    const built = await preloadG1Runtime((progress) => {
      if (loaded || busy) return;
      setLoading("正在预热 MuJoCo", progress.message, "WARMING UP");
      mujocoDiag.textContent = progress.message;
    });
    if (!sim) sim = built;
    if (loaded || busy) return;

    sim = built;
    bindScene(built);
    built.mujoco.mj_forward(built.model, built.data);
    scene?.sync(built);
    scene?.render();
    const ms = Math.round(performance.now() - t0);
    const via = built.source === "mjb" ? "mjb" : "xml";
    setStatus("READY FOR MODEL", "ready");
    logValue.textContent = `MuJoCo 已就绪 (${ms} ms · ${via}) · 拖入 ONNX 即可预览 · 二次打开将走本地缓存`;
    mujocoDiag.textContent = `G1 已就绪 · ${via} · ${ms} ms`;
    setOverlayReady("G1 已就绪", "拖入 .onnx 开始加载策略");
    $("#messageValue").textContent = "物理引擎已预热，可以导入策略";
    setDropEnabled(true);
  } catch (error) {
    if (loaded || busy) return;
    const message = error instanceof Error ? error.message : String(error);
    setStatus("WAITING FOR MODEL", "idle");
    logValue.textContent = `预加载未完成，将在导入策略时重试：${message}`;
    mujocoDiag.textContent = "预热失败 · 导入时重试";
    setOverlayReady("等待 G1 策略", "拖入 .onnx 时会重新初始化 MuJoCo");
    setDropEnabled(true);
  }
}

async function handleFile(file: File): Promise<void> {
  if (busy || !loadPolicy || !createG1Model) return;
  busy = true;
  controller?.setRunning(false);
  playButton.textContent = "▶ PLAY";
  accumulator = 0;
  setStatus("PARSING MODEL", "loading");
  logValue.textContent = `读取 ${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MiB`;
  setLoading("正在加载策略", `${file.name} · 解析 ONNX 与物理绑定…`, "PARSING MODEL");

  let next: PolicyLoadResult | null = null;
  try {
    const t0 = performance.now();
    const alreadyReady = isG1RuntimeReady();
    if (!alreadyReady) setStatus("LOADING POLICY + MUJOCO", "loading");
    else setStatus("LOADING POLICY", "loading");

    const [policyResult, builtSim] = await Promise.all([
      loadPolicy(file),
      createG1Model(G1_POLICY_JOINTS, (progress) => {
        if (alreadyReady) return;
        mujocoDiag.textContent = progress.message;
        const titleEl = overlay.querySelector("strong");
        const detailEl = overlay.querySelector("small");
        if (titleEl) titleEl.textContent = "正在初始化 MuJoCo";
        if (detailEl) detailEl.textContent = progress.message;
      }),
    ]);
    next = policyResult;
    const tPolicy = performance.now();

    rebindPolicyJoints(builtSim, next.contract.metadata.jointNames);
    sim = builtSim;
    bindScene(sim);
    const tSim = performance.now();
    const reusedSim = alreadyReady;

    const previous = loaded;
    loaded = next;
    next = null;
    // Keep the mode the UI currently shows when swapping ONNX.
    const uiMode = selectedPreviewMode();
    controller = new G1ControllerCtor(sim, loaded.session, loaded.contract);
    controller.setMode(uiMode);
    await controller.reset();
    if (previous) await previous.session.release().catch(() => undefined);

    lastTime = performance.now();
    overlay.classList.add("hidden");
    playButton.disabled = false;
    resetButton.disabled = false;
    document.querySelectorAll<HTMLButtonElement>(".mode-switch button").forEach((button) => {
      button.disabled = false;
      button.classList.toggle("active", button.dataset.mode === uiMode);
    });
    $("#contractBadge").textContent = "VALID";
    $("#contractBadge").className = "good";
    $("#diagnosticRows").innerHTML = `<div class="diag-row"><span>MODEL</span><strong title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</strong></div><div class="diag-row"><span>OBSERVATION</span><strong>obs [1 × 154]</strong></div><div class="diag-row"><span>ACTION</span><strong>actions [1 × 29]</strong></div><div class="diag-row"><span>MUJOCO</span><strong>G1 / nq ${sim.model.nq} / dt 5 ms${reusedSim ? " · ready" : " · cold"}</strong></div>`;
    const policyMs = Math.round(tPolicy - t0);
    const simMs = Math.round(tSim - tPolicy);
    const totalMs = Math.round(tSim - t0);
    setStatus("RUNNING", "ready");
    logValue.textContent = reusedSim
      ? `协议校验通过 · MuJoCo 已预热 (rebind ${simMs} ms) · ONNX ${policyMs} ms · 总计 ${totalMs} ms · 模式 ${uiMode}`
      : `协议校验通过 · MuJoCo 冷启动 ${simMs} ms · ONNX ${policyMs} ms · 总计 ${totalMs} ms · 模式 ${uiMode}`;
    updateSnapshot(controller.snapshot());
    // Leave the drop zone so Space no longer re-opens the file picker.
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
    playButton.focus({ preventScroll: true });
  } catch (error) {
    if (next) await next.session.release().catch(() => undefined);
    if (!loaded) {
      if (!isG1RuntimeReady()) {
        disposeG1?.(sim);
        sim = null;
        sceneBound = false;
      }
      controller = null;
    }
    showError(error);
  } finally {
    busy = false;
  }
}

function selectedPreviewMode(): PreviewMode {
  const active = document.querySelector<HTMLButtonElement>(".mode-switch button.active");
  const mode = active?.dataset.mode;
  return mode === "reference" ? "reference" : "policy";
}

function togglePlayback(): void {
  if (!controller || busy) return;
  const running = !controller.isRunning;
  controller.setRunning(running);
  accumulator = 0;
  lastTime = performance.now();
  playButton.textContent = running ? "Ⅱ PAUSE" : "▶ PLAY";
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

function updateSnapshot(snapshot: SimulationSnapshot): void {
  if (!scene || !sim) return;
  const { stats } = snapshot;
  $("#frameValue").textContent = stats.totalFrames === null ? String(stats.frame + 1).padStart(4, "0") : `${String(stats.frame + 1).padStart(4, "0")} / ${stats.totalFrames}`;
  $("#timeValue").textContent = stats.time.toFixed(3);
  $("#heightValue").textContent = stats.pelvisHeight.toFixed(3);
  $("#inferenceValue").textContent = stats.lastInferenceMs === null ? "—" : stats.lastInferenceMs.toFixed(1);
  $("#modeValue").textContent = stats.mode.toUpperCase();
  $("#messageValue").textContent = stats.message;
  scene.sync(sim);
  scene.render();
}

async function tick(now: number): Promise<void> {
  const delta = Math.min(0.1, (now - lastTime) / 1000 || 0);
  lastTime = now;

  // Always bank wall-clock time while playing — including frames spent waiting on
  // an in-flight policy/physics step. Dropping that time made high-res windows
  // (slower rAF / longer GPU frames) play the motion in slow motion.
  if (controller?.isRunning && !busy) {
    accumulator += delta * speed;
  }

  if (controller && controller.isRunning && !busy && !tickInFlight) {
    // Cap backlog so a long stall does not freeze the UI in a catch-up spiral.
    const maxAccum = CONTROL_DT * MAX_CATCH_UP_STEPS;
    if (accumulator > maxAccum) accumulator = maxAccum;

    if (accumulator >= CONTROL_DT) {
      tickInFlight = true;
      try {
        let steps = 0;
        while (accumulator >= CONTROL_DT && steps < MAX_CATCH_UP_STEPS) {
          accumulator -= CONTROL_DT;
          await controller.tick();
          steps += 1;
        }
        // One render after N sim steps keeps GPU cost off the critical path.
        updateSnapshot(controller.snapshot());
      } catch (error) {
        controller.setRunning(false);
        playButton.textContent = "▶ PLAY";
        showError(error);
      } finally {
        tickInFlight = false;
      }
    } else if (sim && scene) {
      // Between control ticks: keep orbit / resize responsive.
      scene.sync(sim);
      scene.render();
    }
  } else if (sim && scene) {
    scene.sync(sim);
    scene.render();
  }
  requestAnimationFrame((time) => { void tick(time); });
}

function wireUi(): void {
  playButton.addEventListener("click", () => {
    togglePlayback();
  });
  resetButton.addEventListener("click", () => {
    if (!controller) return;
    controller.setRunning(false);
    playButton.textContent = "▶ PLAY";
    accumulator = 0;
    void controller.reset().then(() => updateSnapshot(controller!.snapshot()));
  });
  $("#speedInput").addEventListener("input", (event) => {
    speed = Number((event.target as HTMLInputElement).value);
    $("#speedValue").textContent = `${speed}×`;
  });
  document.querySelectorAll<HTMLButtonElement>(".mode-switch button").forEach((button) => button.addEventListener("click", () => {
    if (!controller) return;
    controller.setRunning(false);
    playButton.textContent = "▶ PLAY";
    accumulator = 0;
    const mode = button.dataset.mode as PreviewMode;
    controller.setMode(mode);
    document.querySelectorAll(".mode-switch button").forEach((item) => item.classList.toggle("active", item === button));
    void controller.reset().then(() => updateSnapshot(controller!.snapshot()));
  }));
  dropZone.addEventListener("click", () => {
    if (dropZone.classList.contains("is-disabled")) return;
    fileInput.click();
  });
  // Enter opens the file picker; Space is handled globally for play/pause.
  dropZone.addEventListener("keydown", (event) => {
    const keyEvent = event as KeyboardEvent;
    if (dropZone.classList.contains("is-disabled")) return;
    if (keyEvent.key === "Enter") {
      keyEvent.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) void handleFile(file);
    // Clear value so the same file can be re-selected later.
    fileInput.value = "";
  });
  for (const eventName of ["dragenter", "dragover"]) {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      if (!dropZone.classList.contains("is-disabled")) dropZone.classList.add("dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("dragging");
    });
  }
  dropZone.addEventListener("drop", (event) => {
    if (dropZone.classList.contains("is-disabled")) return;
    const dropEvent = event as DragEvent;
    const file = dropEvent.dataTransfer?.files[0];
    if (file) void handleFile(file);
  });
  // Capture phase + preventDefault so Space does not re-activate the drop zone
  // or double-toggle the focused PLAY button.
  window.addEventListener(
    "keydown",
    (event) => {
      if (event.key !== " " && event.code !== "Space") return;
      if (isTypingTarget(event.target)) return;
      event.preventDefault();
      event.stopPropagation();
      if (!controller || busy || playButton.disabled) return;
      togglePlayback();
    },
    true,
  );
  window.addEventListener("beforeunload", () => {
    disposeG1?.(sim);
    if (loaded) void loaded.session.release();
    scene?.dispose();
  });
}

/** Double-rAF: let the full shell paint before pulling multi‑MB modules. */
function afterFirstPaint(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => resolve());
    });
  });
}

async function boot(): Promise<void> {
  setLoading("正在加载运行时模块", "Three.js · MuJoCo · ONNX Runtime…", "LOADING MODULES");
  await afterFirstPaint();

  try {
    const [policyMod, mujocoMod, sceneMod, controllerMod] = await Promise.all([
      import("./policy/loader"),
      import("./simulation/mujoco"),
      import("./rendering/scene"),
      import("./simulation/controller"),
    ]);

    loadPolicy = policyMod.loadPolicy;
    preloadOrtRuntime = policyMod.preloadOrtRuntime;
    createG1Model = mujocoMod.createG1Model;
    disposeG1 = mujocoMod.disposeG1;
    G1_POLICY_JOINTS = mujocoMod.G1_POLICY_JOINTS;
    isG1RuntimeReady = mujocoMod.isG1RuntimeReady;
    preloadG1Runtime = mujocoMod.preloadG1Runtime;
    rebindPolicyJoints = mujocoMod.rebindPolicyJoints;
    startG1Prefetch = mujocoMod.startG1Prefetch;
    G1ControllerCtor = controllerMod.G1Controller;
    // Ensure prefetch is running even if the early import raced and failed.
    void startG1Prefetch();

    scene = new sceneMod.G1Scene(canvas);
    wireUi();
    requestAnimationFrame((time) => { void tick(time); });
    await warmUpRuntime();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus("BOOT ERROR", "error");
    logValue.textContent = `运行时模块加载失败：${message}`;
    overlay.classList.remove("is-loading");
    overlay.innerHTML = `<span class="crosshair error-mark">×</span><strong>启动失败</strong><small>${escapeHtml(message)}</small>`;
    mujocoDiag.textContent = "启动失败";
  }
}

// Shell is already visible; boot heavy runtime in the background.
void boot();
