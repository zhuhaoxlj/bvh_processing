import * as ort from "onnxruntime-web";
import {
  outputTensor,
  tensorFloatData,
} from "../policy/loader";
import type {
  PolicyContract,
  PreviewMode,
  ReferenceFrame,
  SimulationSnapshot,
} from "../types";
import type { G1Model } from "./mujoco";
import type { DoubleBuffer } from "@mujoco/mujoco";

const CONTROL_DT = 0.02;
const PHYSICS_DT = 0.005;
const STEPS_PER_CONTROL = Math.round(CONTROL_DT / PHYSICS_DT);
const BODY_OBJECT = 1;
// Foot collision geoms are capsules whose lowest support point is approximately
// geom_xpos.z - radius. Reference poses often sit a few mm–cm below z=0 because
// they come from Isaac Lab without MuJoCo contact resolution.
const GROUND_PENETRATION_EPS = 1e-4;

function normalizeQuat(quat: ArrayLike<number>): number[] {
  const norm = Math.hypot(quat[0] ?? 1, quat[1] ?? 0, quat[2] ?? 0, quat[3] ?? 0);
  if (!Number.isFinite(norm) || norm < 1e-8) return [1, 0, 0, 0];
  return [
    (quat[0] ?? 1) / norm,
    (quat[1] ?? 0) / norm,
    (quat[2] ?? 0) / norm,
    (quat[3] ?? 0) / norm,
  ];
}

function quatConjugate(quat: ArrayLike<number>): number[] {
  const normalized = normalizeQuat(quat);
  return [normalized[0], -normalized[1], -normalized[2], -normalized[3]];
}

function quatMultiply(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
  const aw = a[0] ?? 1; const ax = a[1] ?? 0; const ay = a[2] ?? 0; const az = a[3] ?? 0;
  const bw = b[0] ?? 1; const bx = b[1] ?? 0; const by = b[2] ?? 0; const bz = b[3] ?? 0;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

function quatMatrix(quat: ArrayLike<number>): number[] {
  const normalized = normalizeQuat(quat);
  const w = normalized[0];
  const x = normalized[1];
  const y = normalized[2];
  const z = normalized[3];
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
}

function matTransposeVec(matrix: ArrayLike<number>, vector: ArrayLike<number>): number[] {
  return [
    (matrix[0] ?? 0) * (vector[0] ?? 0) + (matrix[3] ?? 0) * (vector[1] ?? 0) + (matrix[6] ?? 0) * (vector[2] ?? 0),
    (matrix[1] ?? 0) * (vector[0] ?? 0) + (matrix[4] ?? 0) * (vector[1] ?? 0) + (matrix[7] ?? 0) * (vector[2] ?? 0),
    (matrix[2] ?? 0) * (vector[0] ?? 0) + (matrix[5] ?? 0) * (vector[1] ?? 0) + (matrix[8] ?? 0) * (vector[2] ?? 0),
  ];
}

function finiteArray(values: ArrayLike<number>): boolean {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) return false;
  }
  return true;
}

export class G1Controller {
  private frame = 0;
  private running = false;
  private mode: PreviewMode = "policy";
  private previousAction = new Float32Array(29);
  private lastAction = new Float32Array(29);
  private lastObservation = new Float32Array(154);
  private motionLength: number | null = null;
  private lastInferenceMs: number | null = null;
  private message = "等待播放";
  private referenceCache = new Map<number, ReferenceFrame>();
  private stepping = false;
  private readonly footBodyIds: number[];

  constructor(
    private readonly sim: G1Model,
    private readonly session: ort.InferenceSession,
    private readonly contract: PolicyContract,
  ) {
    const leftFoot = sim.mujoco.mj_name2id(sim.model, BODY_OBJECT, "left_ankle_roll_link");
    const rightFoot = sim.mujoco.mj_name2id(sim.model, BODY_OBJECT, "right_ankle_roll_link");
    this.footBodyIds = [leftFoot, rightFoot].filter((id) => id >= 0);
  }

  get frameIndex(): number { return this.frame; }
  get isRunning(): boolean { return this.running; }
  get previewMode(): PreviewMode { return this.mode; }
  get totalFrames(): number | null { return this.motionLength; }

  setMode(mode: PreviewMode): void {
    this.mode = mode;
  }

  setRunning(running: boolean): void {
    this.running = running;
    this.message = running ? "播放中" : "已暂停";
  }

  async reset(): Promise<void> {
    this.frame = 0;
    this.previousAction.fill(0);
    this.lastAction.fill(0);
    this.lastObservation.fill(0);
    this.lastInferenceMs = null;
    this.message = "正在重置";
    this.sim.mujoco.mj_resetData(this.sim.model, this.sim.data);
    const reference = await this.reference(0);
    this.applyReference(reference);
    this.message = "已就绪";
  }

  async seek(frame: number): Promise<void> {
    const next = Math.max(0, Math.floor(frame));
    this.frame = next;
    const reference = await this.reference(next);
    this.applyReference(reference);
    this.message = `定位到第 ${next + 1} 帧`;
  }

  async tick(): Promise<SimulationSnapshot> {
    if (!this.running || this.stepping) return this.snapshot();
    this.stepping = true;
    try {
      const start = performance.now();
      if (this.mode === "reference") {
        this.applyReference(await this.reference(this.frame));
      } else {
        await this.policyStep();
      }
      this.lastInferenceMs = performance.now() - start;
      this.frame += 1;
      if (this.motionLength !== null && this.frame >= this.motionLength) {
        this.frame = 0;
        this.previousAction.fill(0);
        this.lastAction.fill(0);
        const reference = await this.reference(0);
        this.applyReference(reference);
        this.message = "动作循环";
      }
      return this.snapshot();
    } finally {
      this.stepping = false;
    }
  }

  snapshot(): SimulationSnapshot {
    const xpos = this.sim.data.xpos as Float64Array;
    const pelvisHeight = xpos[this.sim.pelvisId * 3 + 2] ?? 0;
    return {
      stats: {
        frame: this.frame,
        totalFrames: this.motionLength,
        time: this.sim.data.time,
        pelvisHeight,
        mode: this.mode,
        running: this.running,
        lastInferenceMs: this.lastInferenceMs,
        message: this.message,
      },
      action: this.lastAction,
      observation: this.lastObservation,
    };
  }

  private async reference(frame: number): Promise<ReferenceFrame> {
    const clamped = this.motionLength === null ? Math.max(0, frame) : Math.min(frame, this.motionLength - 1);
    const cached = this.referenceCache.get(clamped);
    if (cached) return cached;
    const result = await this.session.run({
      obs: new ort.Tensor("float32", new Float32Array(154), [1, 154]),
      time_step: new ort.Tensor("float32", Float32Array.of(clamped), [1, 1]),
    });
    const jointPos = tensorFloatData(outputTensor(result, "joint_pos"), "joint_pos");
    const jointVel = tensorFloatData(outputTensor(result, "joint_vel"), "joint_vel");
    const bodyPos = tensorFloatData(outputTensor(result, "body_pos_w"), "body_pos_w");
    const bodyQuat = tensorFloatData(outputTensor(result, "body_quat_w"), "body_quat_w");
    const frameData: ReferenceFrame = {
      jointPos: jointPos.slice(0, 29),
      jointVel: jointVel.slice(0, 29),
      bodyPos: bodyPos.slice(0, 14 * 3),
      bodyQuat: bodyQuat.slice(0, 14 * 4),
    };
    this.referenceCache.set(clamped, frameData);
    if (this.motionLength === null) this.motionLength = this.inferLength(frameData, clamped);
    return frameData;
  }

  private inferLength(lastFrame: ReferenceFrame, index: number): number | null {
    if (index === 0) return null;
    const previous = this.referenceCache.get(index - 1);
    if (!previous) return null;
    for (let item = 0; item < lastFrame.jointPos.length; item += 1) {
      if (lastFrame.jointPos[item] !== previous.jointPos[item]) return null;
    }
    return index;
  }

  private applyReference(reference: ReferenceFrame): void {
    const qpos = this.sim.data.qpos as Float64Array;
    qpos[0] = reference.bodyPos[0] ?? 0;
    qpos[1] = reference.bodyPos[1] ?? 0;
    qpos[2] = reference.bodyPos[2] ?? 0;
    for (let index = 0; index < 4; index += 1) qpos[3 + index] = reference.bodyQuat[index] ?? (index === 0 ? 1 : 0);
    for (let index = 0; index < 29; index += 1) qpos[this.sim.qposIds[index]!] = reference.jointPos[index] ?? 0;
    (this.sim.data.qvel as Float64Array).fill(0);
    this.sim.mujoco.mj_forward(this.sim.model, this.sim.data);
    // Kinematic playback has no contact solver. Lift the free base so the lowest
    // foot support point is not below the ground plane (z = 0). Airborne poses
    // (minZ > 0) are left unchanged.
    this.alignRootToGround();
  }

  private alignRootToGround(): void {
    if (this.footBodyIds.length === 0) return;
    const model = this.sim.model as any;
    const data = this.sim.data as any;
    const geomXpos = data.geom_xpos as Float64Array;
    const geomSize = model.geom_size as Float64Array;
    const geomBodyId = model.geom_bodyid as Int32Array | Uint32Array | number[];
    const geomGroup = model.geom_group as Int32Array | Uint32Array | number[];
    let minSupportZ = Infinity;
    for (let geomId = 0; geomId < model.ngeom; geomId += 1) {
      // Foot sole collisions live in group 3 on the ankle_roll bodies.
      if ((geomGroup[geomId] as number) !== 3) continue;
      const bodyId = geomBodyId[geomId] as number;
      if (!this.footBodyIds.includes(bodyId)) continue;
      const radius = geomSize[geomId * 3] ?? 0.01;
      const centerZ = geomXpos[geomId * 3 + 2] ?? 0;
      minSupportZ = Math.min(minSupportZ, centerZ - radius);
    }
    if (!Number.isFinite(minSupportZ) || minSupportZ >= -GROUND_PENETRATION_EPS) return;
    const qpos = data.qpos as Float64Array;
    qpos[2] = (qpos[2] ?? 0) - minSupportZ;
    this.sim.mujoco.mj_forward(this.sim.model, this.sim.data);
  }

  private async policyStep(): Promise<void> {
    const reference = await this.reference(this.frame);
    const observation = this.observation(reference);
    this.lastObservation.set(observation);
    const result = await this.session.run({
      obs: new ort.Tensor("float32", observation, [1, 154]),
      time_step: new ort.Tensor("float32", Float32Array.of(this.frame), [1, 1]),
    });
    const action = tensorFloatData(outputTensor(result, "actions"), "actions").slice(0, 29);
    if (!finiteArray(action)) throw new Error("策略输出包含 NaN/Inf");
    this.lastAction.set(action);
    const { defaultPos, actionScale, stiffness, damping } = this.contract.metadata;
    const qpos = this.sim.data.qpos as Float64Array;
    const qvel = this.sim.data.qvel as Float64Array;
    const applied = this.sim.data.qfrc_applied as Float64Array;
    for (let step = 0; step < STEPS_PER_CONTROL; step += 1) {
      applied.fill(0);
      for (let index = 0; index < 29; index += 1) {
        const qposId = this.sim.qposIds[index]!;
        const dofId = this.sim.dofIds[index]!;
        const target = defaultPos[index]! + actionScale[index]! * action[index]!;
        const torque = stiffness[index]! * (target - qpos[qposId]!) - damping[index]! * qvel[dofId]!;
        applied[dofId] = Math.max(-this.sim.forceLimits[index]!, Math.min(this.sim.forceLimits[index]!, torque));
      }
      this.sim.mujoco.mj_step(this.sim.model, this.sim.data);
    }
    this.previousAction.set(action);
  }

  private observation(reference: ReferenceFrame): Float32Array {
    const terms: Record<string, number[]> = {};
    terms.command = [...reference.jointPos, ...reference.jointVel];
    const torsoPos = this.bodyPosition(this.sim.torsoId);
    const torsoQuat = this.bodyQuaternion(this.sim.torsoId);
    const anchorPos = reference.bodyPos.slice(this.contract.anchorBodyIndex * 3, this.contract.anchorBodyIndex * 3 + 3);
    const anchorQuat = reference.bodyQuat.slice(this.contract.anchorBodyIndex * 4, this.contract.anchorBodyIndex * 4 + 4);
    const torsoRotation = quatMatrix(torsoQuat);
    const relativeRotation = quatMatrix(quatMultiply(quatConjugate(torsoQuat), anchorQuat));
    // Isaac Lab flattens rotation[:, :2] in row-major order: first two columns,
    // not the first six entries of the full row-major 3x3 matrix.
    terms.motion_anchor_ori_b = [
      relativeRotation[0], relativeRotation[1],
      relativeRotation[3], relativeRotation[4],
      relativeRotation[6], relativeRotation[7],
    ];
    terms.motion_anchor_pos_b = matTransposeVec(torsoRotation, [anchorPos[0]! - torsoPos[0]!, anchorPos[1]! - torsoPos[1]!, anchorPos[2]! - torsoPos[2]!]);
    const velocityBuffer: DoubleBuffer = new this.sim.mujoco.DoubleBuffer(6);
    try {
      this.sim.mujoco.mj_objectVelocity(this.sim.model, this.sim.data, BODY_OBJECT, this.sim.pelvisId, velocityBuffer, 1);
      const velocity = velocityBuffer.GetView() as Float64Array;
      terms.base_ang_vel = Array.from(velocity.slice(0, 3));
      terms.base_lin_vel = Array.from(velocity.slice(3, 6));
    } finally {
      velocityBuffer.delete();
    }
    terms.joint_pos = this.sim.qposIds.map((id, index) => qposValue(this.sim.data.qpos, id) - this.contract.metadata.defaultPos[index]!);
    terms.joint_vel = this.sim.dofIds.map((id) => qposValue(this.sim.data.qvel, id));
    terms.actions = Array.from(this.previousAction);
    const values = this.contract.metadata.observationNames.flatMap((name) => {
      const value = terms[name];
      if (!value) throw new Error(`不支持的 observation term: ${name}`);
      return value;
    });
    if (values.length !== 154 || !finiteArray(values)) throw new Error("observation shape 或数值无效");
    return Float32Array.from(values);
  }

  private bodyPosition(bodyId: number): number[] {
    const values = this.sim.data.xpos as Float64Array;
    return [values[bodyId * 3]!, values[bodyId * 3 + 1]!, values[bodyId * 3 + 2]!];
  }

  private bodyQuaternion(bodyId: number): number[] {
    const values = this.sim.data.xquat as Float64Array;
    return [values[bodyId * 4]!, values[bodyId * 4 + 1]!, values[bodyId * 4 + 2]!, values[bodyId * 4 + 3]!];
  }
}

function qposValue(values: unknown, index: number): number {
  return (values as ArrayLike<number>)[index] ?? 0;
}
