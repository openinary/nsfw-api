import { readFileSync } from "node:fs";
import path from "node:path";
import sharp from "sharp";
import { ImageDecodeError } from "./classifier.ts";
import { createSession, loadOrt, type SessionOptions } from "./ort.ts";

// MobileCLIP-S0 preprocessing: shortest edge to 256 (bilinear), centre crop,
// pixels scaled to 0..1 with no mean/std normalisation.
const MODEL = "Xenova/mobileclip_s0"; // must match scripts/prepare-clip.ts
const SIZE = 256;
const LOGIT_SCALE = 100; // MobileCLIP's learned temperature

export interface CategoryScore {
  score: number;
  /** The category's prompt CLIP matched best. */
  match: string | null;
}

export interface ClipClassifier {
  classify(buffer: Buffer): Promise<Record<string, CategoryScore>>;
}

/** models/clip/prompts.json, written by scripts/prepare-clip.ts. */
export interface PromptsFile {
  model: string;
  revision: string;
  variant: string;
  dim: number;
  prompts: { category: string; text: string; embedding: number[] }[];
}

async function pixels(buffer: Buffer) {
  let data: Buffer;
  try {
    ({ data } = await sharp(buffer, { failOn: "error", limitInputPixels: 100_000_000 })
      .rotate()
      .flatten({ background: "#ffffff" })
      .resize(SIZE, SIZE, { fit: "cover", kernel: "linear" })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true }));
  } catch (err) {
    throw new ImageDecodeError(err);
  }
  const plane = SIZE * SIZE;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) out[c * plane + i] = data[i * 3 + c] / 255;
  }
  return out;
}

/**
 * @param opts.categories  enabled categories (subset of prompts.json)
 * @param opts.modelFile  vision model file inside modelDir
 * @param opts.sessionOptions  extra ONNX Runtime session options
 */
export async function createClipClassifier({
  categories,
  modelDir = "models/clip",
  threads = 1,
  modelFile = "vision.onnx",
  sessionOptions = {},
}: {
  categories: string[];
  modelDir?: string;
  threads?: number;
  modelFile?: string;
  sessionOptions?: SessionOptions;
}) {
  const ort = await loadOrt();
  const { model, prompts, dim, variant }: PromptsFile = JSON.parse(
    readFileSync(path.join(modelDir, "prompts.json"), "utf8"),
  );
  if (model !== MODEL) {
    throw new Error(`${modelDir}/prompts.json was made for ${model}, not ${MODEL}: run "npm run prepare-models" again`);
  }

  const known = new Set(prompts.map((p) => p.category));
  for (const c of categories) {
    if (!known.has(c) || c === "other") throw new Error(`Unknown CLIP category "${c}"`);
  }
  // Only enabled categories compete with the other prompts in the softmax.
  const active = prompts.filter((p) => p.category === "other" || categories.includes(p.category));
  const matrix = new Float32Array(active.length * dim);
  active.forEach((p, i) => matrix.set(p.embedding, i * dim));

  const session = await createSession(path.join(modelDir, modelFile), { threads, ...sessionOptions });

  async function classify(buffer: Buffer): Promise<Record<string, CategoryScore>> {
    const input = new ort.Tensor("float32", await pixels(buffer), [1, 3, SIZE, SIZE]);
    const { image_embeds } = await session.run({ pixel_values: input });
    const img = image_embeds.data as Float32Array;
    const norm = Math.hypot(...img);

    const logits = new Float32Array(active.length);
    let max = -Infinity;
    for (let i = 0; i < active.length; i++) {
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += img[d] * matrix[i * dim + d];
      logits[i] = (LOGIT_SCALE * dot) / norm;
      if (logits[i] > max) max = logits[i];
    }
    let total = 0;
    for (let i = 0; i < logits.length; i++) total += logits[i] = Math.exp(logits[i] - max);

    const scores: Record<string, number> = Object.fromEntries(categories.map((c) => [c, 0]));
    const best: Record<string, { p: number; text: string }> = {};
    for (let i = 0; i < active.length; i++) {
      const { category, text } = active[i];
      if (category === "other") continue;
      const p = logits[i] / total;
      scores[category] += p;
      if (!best[category] || p > best[category].p) best[category] = { p, text };
    }
    return Object.fromEntries(
      categories.map((c) => [c, { score: scores[c], match: best[c]?.text ?? null }]),
    );
  }

  // Warm-up so the first request isn't slow.
  await classify(await sharp({ create: { width: SIZE, height: SIZE, channels: 3, background: "#808080" } }).png().toBuffer());

  return { classify, categories, variant, dispose: () => session.release() };
}
