import loadMujoco from "@mujoco/mujoco";
import mujocoWasmUrl from "@mujoco/mujoco/mujoco.wasm?url";
import type { MjData, MjModel, MjVFS, MainModule } from "@mujoco/mujoco";
import { assetUrl, cacheSet, fetchCached, type CacheSource } from "./cache";

export type MujocoModule = MainModule;

export interface G1Model {
  mujoco: MujocoModule;
  model: MjModel;
  data: MjData;
  jointIds: number[];
  qposIds: number[];
  dofIds: number[];
  forceLimits: number[];
  pelvisId: number;
  torsoId: number;
  /** Present only when the model was built from XML+meshes (fallback path). */
  vfs: MjVFS | null;
  source: "mjb" | "xml";
}

const JOINT_OBJECT = 3;
const BODY_OBJECT = 1;
const MJB_NAME = "g1.mjb";
const META_URL = assetUrl("robots/g1/g1.asset.json");
const MJB_URL = assetUrl("robots/g1/g1.mjb");
const WASM_CACHE_PREFIX = "mujoco-wasm@3.11.0";

/** Module-level caches so reloading a policy does not re-download WASM/MJB. */
let mujocoPromise: Promise<MujocoModule> | null = null;
let sharedSimPromise: Promise<G1Model> | null = null;
let sharedSim: G1Model | null = null;
let prefetchPromise: Promise<PrefetchState> | null = null;
let prefetchState: PrefetchState | null = null;

interface G1AssetMeta {
  version: string;
  mjbBytes: number;
  mjbFile?: string;
}

interface PrefetchState {
  meta: G1AssetMeta | null;
  wasm: ArrayBuffer | null;
  mjb: ArrayBuffer | null;
  wasmSource: CacheSource;
  mjbSource: CacheSource;
}

export type G1PreloadStage = "prefetch" | "wasm" | "model" | "ready";

export type G1PreloadProgress = {
  stage: G1PreloadStage;
  message: string;
};

type ProgressFn = ((progress: G1PreloadProgress) => void) | undefined;

async function loadMeta(): Promise<G1AssetMeta | null> {
  try {
    const response = await fetch(META_URL);
    if (!response.ok) return null;
    return (await response.json()) as G1AssetMeta;
  } catch {
    return null;
  }
}

/**
 * Start downloading WASM + precompiled MJB as soon as the shell paints.
 * Safe to call before the rest of the runtime finishes importing.
 */
export function startG1Prefetch(): Promise<PrefetchState> {
  if (prefetchPromise) return prefetchPromise;
  prefetchPromise = (async () => {
    const meta = await loadMeta();
    const wasmKey = WASM_CACHE_PREFIX;
    const mjbKey = meta?.version ? `g1-mjb:${meta.version}` : "g1-mjb:latest";

    const [wasmResult, mjbResult] = await Promise.all([
      fetchCached(wasmKey, mujocoWasmUrl).catch(() => ({ buffer: null as ArrayBuffer | null, source: "none" as CacheSource })),
      fetchCached(mjbKey, MJB_URL).catch(() => ({ buffer: null as ArrayBuffer | null, source: "none" as CacheSource })),
    ]);

    // If meta was missing initially, still try to cache under a stable key after download.
    if (mjbResult.buffer && meta?.version) {
      void cacheSet(`g1-mjb:${meta.version}`, mjbResult.buffer);
    }

    const state: PrefetchState = {
      meta,
      wasm: wasmResult.buffer,
      mjb: mjbResult.buffer,
      wasmSource: wasmResult.source,
      mjbSource: mjbResult.source,
    };
    prefetchState = state;
    return state;
  })().catch((error) => {
    prefetchPromise = null;
    throw error;
  });
  return prefetchPromise;
}

async function getMujoco(onProgress?: ProgressFn): Promise<MujocoModule> {
  if (!mujocoPromise) {
    mujocoPromise = (async () => {
      onProgress?.({ stage: "wasm", message: "初始化 MuJoCo WASM…" });
      const prefetched = prefetchState ?? (await startG1Prefetch().catch(() => null));
      if (prefetched?.wasm) {
        const sourceHint =
          prefetched.wasmSource === "idb" ? "缓存" : prefetched.wasmSource === "network" ? "网络" : "";
        onProgress?.({
          stage: "wasm",
          message: sourceHint ? `加载 MuJoCo WASM（${sourceHint}）…` : "加载 MuJoCo WASM…",
        });
        // Copy so Emscripten cannot detach the cached/prefetched buffer.
        return loadMujoco({ wasmBinary: prefetched.wasm.slice(0) });
      }
      onProgress?.({ stage: "wasm", message: "下载并初始化 MuJoCo WASM…" });
      return loadMujoco();
    })().catch((error) => {
      mujocoPromise = null;
      throw error;
    });
  }
  return mujocoPromise;
}

function bindPolicyJoints(
  mujoco: MujocoModule,
  model: MjModel,
  policyJointNames: readonly string[],
): Pick<G1Model, "jointIds" | "qposIds" | "dofIds" | "forceLimits"> {
  if (policyJointNames.length !== 29) throw new Error("策略 metadata 必须包含 29 个关节");
  const jointIds = policyJointNames.map((name) => mujoco.mj_name2id(model, JOINT_OBJECT, name));
  if (jointIds.some((id) => id < 0)) throw new Error("G1 MJCF 缺少策略关节");
  const qposIds = jointIds.map((id) => model.jnt_qposadr[id] as number);
  const dofIds = jointIds.map((id) => model.jnt_dofadr[id] as number);
  const forceLimits = jointIds.map((id, index) => {
    const limit = (model.jnt_actfrcrange as Float64Array)[id * 2 + 1];
    if (limit === undefined) throw new Error(`缺少关节力矩限制: ${policyJointNames[index] ?? id}`);
    return Math.abs(limit);
  });
  return { jointIds, qposIds, dofIds, forceLimits };
}

/** Rebind joint index maps when swapping policies without rebuilding the MJCF. */
export function rebindPolicyJoints(sim: G1Model, policyJointNames: readonly string[]): void {
  const bound = bindPolicyJoints(sim.mujoco, sim.model, policyJointNames);
  sim.jointIds = bound.jointIds;
  sim.qposIds = bound.qposIds;
  sim.dofIds = bound.dofIds;
  sim.forceLimits = bound.forceLimits;
}

async function loadMjbModel(
  mujoco: MujocoModule,
  mjb: ArrayBuffer,
  source: CacheSource,
  onProgress?: ProgressFn,
): Promise<{ model: MjModel; vfs: MjVFS }> {
  const hint = source === "idb" ? "缓存" : source === "network" ? "网络" : "内存";
  onProgress?.({ stage: "model", message: `加载预编译 G1 模型（${hint}）…` });
  const vfs = new mujoco.MjVFS();
  vfs.addBuffer(MJB_NAME, new Uint8Array(mjb.slice(0)));
  const model = mujoco.MjModel.mj_loadModel(MJB_NAME, vfs);
  return { model, vfs };
}

/** Fallback: XML + individual (decimated) STLs when MJB is missing. */
async function loadXmlModel(mujoco: MujocoModule, onProgress?: ProgressFn): Promise<{ model: MjModel; vfs: MjVFS }> {
  onProgress?.({ stage: "model", message: "回退：加载 G1 XML 与网格…" });
  const [xmlResponse, ...meshResponses] = await Promise.all([
    fetch(assetUrl("robots/g1/g1.xml")),
    ...G1_MESHES.map((file) => fetch(assetUrl(`robots/g1/meshes/${file}`))),
  ]);
  if (!xmlResponse.ok) throw new Error(`无法加载 G1 XML (${xmlResponse.status})`);
  if (meshResponses.some((response) => !response?.ok)) throw new Error("无法加载 G1 STL 资源");

  const [xmlText, ...meshBuffers] = await Promise.all([
    xmlResponse.text(),
    ...meshResponses.map(async (response, index) => {
      if (!response) throw new Error(`缺少 G1 STL 资源: ${G1_MESHES[index]}`);
      return new Uint8Array(await response.arrayBuffer());
    }),
  ]);

  const xml = xmlText.replace('meshdir="../meshes/g1"', 'meshdir="."');
  const vfs = new mujoco.MjVFS();
  for (const [index, file] of G1_MESHES.entries()) {
    const bytes = meshBuffers[index];
    if (!bytes) throw new Error(`缺少 G1 STL 资源: ${file}`);
    vfs.addBuffer(file, bytes);
  }
  onProgress?.({ stage: "model", message: "编译 G1 MJCF…" });
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  const model = mujoco.MjModel.from_xml_string(xml, vfs);
  return { model, vfs };
}

async function buildG1Model(
  policyJointNames: readonly string[],
  onProgress?: ProgressFn,
): Promise<G1Model> {
  onProgress?.({ stage: "prefetch", message: "等待 WASM / MJB 预取…" });
  const prefetched = prefetchState ?? (await startG1Prefetch().catch(() => null));
  const mujoco = await getMujoco(onProgress);

  let model: MjModel;
  let vfs: MjVFS | null = null;
  let source: "mjb" | "xml" = "mjb";

  if (prefetched?.mjb && prefetched.mjb.byteLength > 0) {
    const loaded = await loadMjbModel(mujoco, prefetched.mjb, prefetched.mjbSource, onProgress);
    model = loaded.model;
    // mj_loadModel keeps mesh data inside the model; VFS can be dropped after load.
    loaded.vfs.delete();
    vfs = null;
    source = "mjb";
  } else {
    // Last-chance direct fetch of MJB before XML fallback.
    try {
      const meta = prefetched?.meta ?? (await loadMeta());
      const key = meta?.version ? `g1-mjb:${meta.version}` : "g1-mjb:latest";
      const { buffer, source: mjbSource } = await fetchCached(key, MJB_URL);
      const loaded = await loadMjbModel(mujoco, buffer, mjbSource, onProgress);
      model = loaded.model;
      loaded.vfs.delete();
      vfs = null;
      source = "mjb";
    } catch {
      const loaded = await loadXmlModel(mujoco, onProgress);
      model = loaded.model;
      vfs = loaded.vfs;
      source = "xml";
    }
  }

  const data = new mujoco.MjData(model);
  model.opt.timestep = 0.005;

  const bound = bindPolicyJoints(mujoco, model, policyJointNames);
  const pelvisId = mujoco.mj_name2id(model, BODY_OBJECT, "pelvis");
  const torsoId = mujoco.mj_name2id(model, BODY_OBJECT, "torso_link");
  if (pelvisId < 0 || torsoId < 0) throw new Error("G1 MJCF 缺少 pelvis 或 torso_link");
  onProgress?.({
    stage: "ready",
    message: source === "mjb" ? "MuJoCo 就绪（预编译模型）" : "MuJoCo 就绪（XML 回退）",
  });
  return {
    mujoco,
    model,
    data,
    ...bound,
    pelvisId,
    torsoId,
    vfs,
    source,
  };
}

/**
 * Returns the shared compiled G1 sim, building it once on first call.
 * Subsequent calls rebind joint maps without reloading WASM/MJB.
 */
export async function createG1Model(
  policyJointNames: readonly string[],
  onProgress?: ProgressFn,
): Promise<G1Model> {
  if (sharedSim) {
    rebindPolicyJoints(sharedSim, policyJointNames);
    onProgress?.({ stage: "ready", message: "MuJoCo 已预热" });
    return sharedSim;
  }
  if (!sharedSimPromise) {
    sharedSimPromise = buildG1Model(G1_POLICY_JOINTS, onProgress)
      .then((sim) => {
        sharedSim = sim;
        return sim;
      })
      .catch((error) => {
        sharedSimPromise = null;
        throw error;
      });
  } else if (onProgress) {
    onProgress({ stage: "prefetch", message: "等待已在进行的 MuJoCo 预热…" });
  }
  const sim = await sharedSimPromise;
  rebindPolicyJoints(sim, policyJointNames);
  onProgress?.({ stage: "ready", message: "MuJoCo 就绪" });
  return sim;
}

/**
 * Start loading MuJoCo WASM + G1 model immediately (e.g. on page open)
 * so the first policy drop only needs ONNX + rebind.
 */
export function preloadG1Runtime(onProgress?: ProgressFn): Promise<G1Model> {
  return createG1Model(G1_POLICY_JOINTS, onProgress);
}

/** True when the shared sim is already compiled and ready to rebind. */
export function isG1RuntimeReady(): boolean {
  return sharedSim !== null;
}

export function disposeG1(sim: G1Model | null): void {
  if (!sim) return;
  if (sim === sharedSim) {
    sharedSim = null;
    sharedSimPromise = null;
  }
  sim.data.delete();
  sim.model.delete();
  sim.vfs?.delete();
}

export const G1_POLICY_JOINTS = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
];

const G1_MESHES = [
  "pelvis.STL", "pelvis_contour_link.STL", "left_hip_pitch_link.STL", "left_hip_roll_link.STL", "left_hip_yaw_link.STL", "left_knee_link.STL",
  "left_ankle_pitch_link.STL", "left_ankle_roll_link.STL", "right_hip_pitch_link.STL", "right_hip_roll_link.STL", "right_hip_yaw_link.STL", "right_knee_link.STL",
  "right_ankle_pitch_link.STL", "right_ankle_roll_link.STL", "waist_yaw_link_rev_1_0.STL", "waist_roll_link_rev_1_0.STL", "torso_link_rev_1_0.STL",
  "logo_link.STL", "head_link.STL", "left_shoulder_pitch_link.STL", "left_shoulder_roll_link.STL", "left_shoulder_yaw_link.STL", "left_elbow_link.STL",
  "left_wrist_roll_link.STL", "left_wrist_pitch_link.STL", "left_wrist_yaw_link.STL", "left_rubber_hand.STL", "right_shoulder_pitch_link.STL", "right_shoulder_roll_link.STL",
  "right_shoulder_yaw_link.STL", "right_elbow_link.STL", "right_wrist_roll_link.STL", "right_wrist_pitch_link.STL", "right_wrist_yaw_link.STL", "right_rubber_hand.STL",
];
