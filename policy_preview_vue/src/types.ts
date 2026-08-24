export type PreviewMode = "reference" | "policy";

export interface PolicyMetadata {
  runPath?: string;
  jointNames: string[];
  stiffness: number[];
  damping: number[];
  defaultPos: number[];
  observationNames: string[];
  actionScale: number[];
}

export interface TensorDescriptor {
  name: string;
  shape: readonly (number | string)[];
  type: string;
}

export interface PolicyContract {
  metadata: PolicyMetadata;
  inputs: TensorDescriptor[];
  outputs: TensorDescriptor[];
  observationSize: number;
  actionSize: number;
  motionLength: number | null;
  bodyNames: string[];
  anchorBodyIndex: number;
}

export interface PolicyLoadResult {
  contract: PolicyContract;
  session: import("onnxruntime-web").InferenceSession;
  modelBytes: Uint8Array;
  fileName: string;
  fileSize: number;
}

export interface ReferenceFrame {
  jointPos: Float32Array;
  jointVel: Float32Array;
  bodyPos: Float32Array;
  bodyQuat: Float32Array;
}

export interface SimulationStats {
  frame: number;
  totalFrames: number | null;
  time: number;
  pelvisHeight: number;
  mode: PreviewMode;
  running: boolean;
  lastInferenceMs: number | null;
  message: string;
}

export interface SimulationSnapshot {
  stats: SimulationStats;
  action?: Float32Array;
  observation?: Float32Array;
}
