import type { InferenceSession } from "onnxruntime-node";

// ONNX Runtime (1.30+) ships usage telemetry; a moderation API should not phone home.
process.env.ORT_DISABLE_TELEMETRY ??= "1";

type Ort = typeof import("onnxruntime-node");

let ort: Ort | undefined;
export async function loadOrt(): Promise<Ort> {
  ort ??= await import("onnxruntime-node");
  return ort;
}

export type SessionOptions = InferenceSession.SessionOptions & { threads?: number };

/**
 * One single-threaded session per model: parallelism comes from concurrent
 * requests (MAX_CONCURRENCY), and extra intra-op threads mostly add overhead
 * on Railway's shared vCPUs.
 */
export async function createSession(file: string, { threads = 1, ...options }: SessionOptions = {}) {
  const { InferenceSession } = await loadOrt();
  return InferenceSession.create(file, {
    intraOpNumThreads: threads,
    interOpNumThreads: 1,
    graphOptimizationLevel: "all",
    ...options,
  });
}
