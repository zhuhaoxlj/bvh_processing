import * as ort from "onnxruntime-web";
import ortWasmMjsUrl from "../ort/ort-wasm-simd-threaded.jsep.mjs?url";
import ortWasmUrl from "../ort/ort-wasm-simd-threaded.jsep.wasm?url";
import { readPolicyMetadata } from "./metadata";
import type {
  PolicyContract,
  PolicyLoadResult,
  TensorDescriptor,
} from "../types";

const REQUIRED_INPUTS = ["obs", "time_step"] as const;
const REQUIRED_OUTPUTS = [
  "actions",
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
] as const;

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = { mjs: ortWasmMjsUrl, wasm: ortWasmUrl };

let ortPreloadStarted = false;

/**
 * Warm the browser HTTP cache for ORT WASM so the first InferenceSession.create
 * does not wait on a cold 26 MiB download.
 */
export function preloadOrtRuntime(): void {
  if (ortPreloadStarted) return;
  ortPreloadStarted = true;
  void fetch(ortWasmUrl, { credentials: "same-origin", mode: "cors" }).catch(() => undefined);
  void fetch(ortWasmMjsUrl, { credentials: "same-origin", mode: "cors" }).catch(() => undefined);
}

function descriptors(
  names: readonly string[],
  metadata: readonly ort.InferenceSession.ValueMetadata[],
): TensorDescriptor[] {
  return names.map((name, index) => {
    const item = metadata[index];
    if (!item) throw new Error(`无法读取 ONNX tensor metadata: ${name}`);
    if (!item.isTensor) throw new Error(`${name} 不是 tensor 输出`);
    return { name, shape: item.shape, type: item.type };
  });
}

function staticDimension(shape: readonly (number | string)[], index: number): number {
  const value = shape[index];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`ONNX shape ${JSON.stringify(shape)} 不是静态 shape`);
  }
  return value;
}

function requireNames(actual: readonly string[], required: readonly string[], kind: string): void {
  const missing = required.filter((name) => !actual.includes(name));
  if (missing.length) throw new Error(`${kind} 缺少: ${missing.join(", ")}`);
}

export async function loadPolicy(file: File): Promise<PolicyLoadResult> {
  if (!file.name.toLowerCase().endsWith(".onnx")) {
    throw new Error("请选择 .onnx 策略文件");
  }

  const bytes = new Uint8Array(await file.arrayBuffer());
  // Metadata parse (protobuf) and ORT session create both walk the full model;
  // overlap them so large embedded-motion ONNX files load faster.
  const [metadata, session] = await Promise.all([
    Promise.resolve().then(() => readPolicyMetadata(bytes)),
    ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    }),
  ]);

  requireNames(session.inputNames, REQUIRED_INPUTS, "模型输入");
  requireNames(session.outputNames, REQUIRED_OUTPUTS, "模型输出");

  const inputDescriptors = descriptors(session.inputNames, session.inputMetadata);
  const outputDescriptors = descriptors(session.outputNames, session.outputMetadata);
  const obsDescriptor = inputDescriptors.find((item) => item.name === "obs");
  const timeDescriptor = inputDescriptors.find((item) => item.name === "time_step");
  const actionDescriptor = outputDescriptors.find((item) => item.name === "actions");
  if (!obsDescriptor || !timeDescriptor || !actionDescriptor) {
    throw new Error("模型输入输出协议不完整");
  }

  if (staticDimension(obsDescriptor.shape, 0) !== 1 || staticDimension(obsDescriptor.shape, 1) !== 154) {
    throw new Error(`当前 G1 策略需要 obs [1, 154]，实际为 ${JSON.stringify(obsDescriptor.shape)}`);
  }
  if (staticDimension(timeDescriptor.shape, 0) !== 1 || staticDimension(timeDescriptor.shape, 1) !== 1) {
    throw new Error(`time_step 必须为 [1, 1]，实际为 ${JSON.stringify(timeDescriptor.shape)}`);
  }
  if (staticDimension(actionDescriptor.shape, 0) !== 1 || staticDimension(actionDescriptor.shape, 1) !== 29) {
    throw new Error(`当前 G1 策略需要 actions [1, 29]，实际为 ${JSON.stringify(actionDescriptor.shape)}`);
  }

  const bodyPosDescriptor = outputDescriptors.find((item) => item.name === "body_pos_w");
  const bodyCount = bodyPosDescriptor ? staticDimension(bodyPosDescriptor.shape, 1) : 0;
  if (bodyCount !== 14) throw new Error(`当前 G1 策略需要 14 个参考 body，实际为 ${bodyCount}`);
  const contract: PolicyContract = {
    metadata,
    inputs: inputDescriptors,
    outputs: outputDescriptors,
    observationSize: 154,
    actionSize: 29,
    motionLength: null,
    bodyNames: [
      "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
      "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
      "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
      "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
    ],
    anchorBodyIndex: 7,
  };
  return { contract, session, modelBytes: bytes, fileName: file.name, fileSize: file.size };
}

export function outputTensor(result: ort.InferenceSession.ReturnType, name: string): ort.Tensor {
  const output = result[name];
  if (!output || !(output instanceof ort.Tensor)) throw new Error(`推理缺少输出 ${name}`);
  return output;
}

export function tensorFloatData(tensor: ort.Tensor, name: string): Float32Array {
  if (!(tensor.data instanceof Float32Array)) throw new Error(`${name} 必须输出 float32`);
  return tensor.data;
}
