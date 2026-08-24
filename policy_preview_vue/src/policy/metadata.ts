import { onnx } from "onnx-proto";
import type { PolicyMetadata } from "../types";

function parseCsv(value: string | null | undefined, name: string): string[] {
  if (!value) throw new Error(`ONNX metadata 缺少 ${name}`);
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseNumbers(value: string | null | undefined, name: string): number[] {
  const values = parseCsv(value, name).map(Number);
  if (values.some((item) => !Number.isFinite(item))) {
    throw new Error(`ONNX metadata ${name} 包含无效数字`);
  }
  return values;
}

export function readPolicyMetadata(bytes: Uint8Array): PolicyMetadata {
  const model = onnx.ModelProto.decode(bytes);
  const entries = new Map(
    model.metadataProps.map((entry) => [entry.key, entry.value] as const),
  );

  const metadata: PolicyMetadata = {
    runPath: entries.get("run_path") ?? undefined,
    jointNames: parseCsv(entries.get("joint_names"), "joint_names"),
    stiffness: parseNumbers(entries.get("joint_stiffness"), "joint_stiffness"),
    damping: parseNumbers(entries.get("joint_damping"), "joint_damping"),
    defaultPos: parseNumbers(entries.get("default_joint_pos"), "default_joint_pos"),
    observationNames: parseCsv(entries.get("observation_names"), "observation_names"),
    actionScale: parseNumbers(entries.get("action_scale"), "action_scale"),
  };

  if (metadata.jointNames.length !== 29) {
    throw new Error(`当前预览器需要 G1 29DoF，模型包含 ${metadata.jointNames.length} 个关节`);
  }
  for (const [name, values] of Object.entries({
    joint_stiffness: metadata.stiffness,
    joint_damping: metadata.damping,
    default_joint_pos: metadata.defaultPos,
    action_scale: metadata.actionScale,
  })) {
    if (values.length !== 29) throw new Error(`metadata ${name} 必须包含 29 个值`);
  }

  return metadata;
}

export function inferMotionLength(
  outputs: readonly { data: unknown }[],
  jointOutputIndex: number,
): number | null {
  const data = outputs[jointOutputIndex]?.data;
  if (!data || !(data instanceof Float32Array || data instanceof Float64Array)) return null;
  return data.length >= 29 ? 1 : null;
}
