// ONNX Runtime (1.30+) ships usage telemetry; a moderation API should not phone home.
process.env.ORT_DISABLE_TELEMETRY ??= "1";

let ort;
export async function loadOrt() {
  ort ??= await import("onnxruntime-node");
  return ort;
}

/**
 * One single-threaded session per model: parallelism comes from concurrent
 * requests (MAX_CONCURRENCY), and extra intra-op threads mostly add overhead
 * on Railway's shared vCPUs.
 */
export async function createSession(file, { threads = 1, ...options } = {}) {
  const { InferenceSession } = await loadOrt();
  return InferenceSession.create(file, {
    intraOpNumThreads: threads,
    interOpNumThreads: 1,
    graphOptimizationLevel: "all",
    ...options,
  });
}
