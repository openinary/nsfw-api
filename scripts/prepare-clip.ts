// Build-time step for the CLIP module:
//  1. downloads MobileCLIP-S0's vision model at a pinned revision, in the
//     precision picked by CLIP_VARIANT (fp32 by default, see VARIANTS), with its license,
//  2. turns every prompt of src/clip-categories.ts into an embedding with the
//     text model, so the text model never ships in the image.
// Output: models/clip/vision.onnx, models/clip/prompts.json, models/clip/LICENSE
import { createWriteStream, existsSync, readFileSync } from "node:fs";
import { mkdir, rename, writeFile } from "node:fs/promises";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { AutoTokenizer, CLIPTextModelWithProjection, env } from "@huggingface/transformers";
import type { PromptsFile } from "../src/clip.ts";
import { CATEGORIES, OTHER } from "../src/clip-categories.ts";

// Apple's MobileCLIP-S0 converted to ONNX by Xenova. This revision ships the
// weights under the Apple Sample Code License of the July 2024 release, copied
// next to the model. Apple has since (August 2025) moved MobileCLIP's official
// weights to a research-only license: see the README before reusing this.
export const MODEL = "Xenova/mobileclip_s0";
export const REVISION = "757d59c9c6870a76a4b0306f05f5061bca15c39f";
const OUT = "models/clip";

// Measured on the test images (weapon, drugs, nudity, drawings), 1 vCPU:
const VARIANTS: Record<string, { file: string; note: string }> = {
  fp32: { file: "vision_model.onnx", note: "46 MB, fastest, lowest RAM" },
  fp16: { file: "vision_model_fp16.onnx", note: "23 MB, same verdicts, ~5% slower, ~+10 MB RAM" },
  // The int8 export ("vision_model_quantized.onnx", 12 MB) is left out on purpose:
  // it missed both the weapon and the drugs pictures.
};
const variant = process.env.CLIP_VARIANT || "fp32";
if (!VARIANTS[variant]) throw new Error(`CLIP_VARIANT must be one of ${Object.keys(VARIANTS).join(", ")}`);

await mkdir(OUT, { recursive: true });
env.cacheDir = "models/.cache";

async function download(file: string, dest: string) {
  const res = await fetch(`https://huggingface.co/${MODEL}/resolve/${REVISION}/${file}`);
  if (!res.ok || !res.body) throw new Error(`Download of ${file} failed: HTTP ${res.status}`);
  await pipeline(Readable.fromWeb(res.body), createWriteStream(dest + ".part"));
  await rename(dest + ".part", dest);
}

const vision = `${OUT}/vision.onnx`;
const stamp = `${OUT}/vision.variant`;
const id = `${MODEL}@${REVISION}:${variant}`;
if (!existsSync(vision) || !existsSync(stamp) || readFileSync(stamp, "utf8") !== id) {
  console.log(`Downloading MobileCLIP-S0 vision model, ${variant} (${VARIANTS[variant].note})…`);
  await download(`onnx/${VARIANTS[variant].file}`, vision);
  await writeFile(stamp, id);
}
if (!existsSync(`${OUT}/LICENSE`)) await download("LICENSE", `${OUT}/LICENSE`);

const entries = [
  ...Object.entries(CATEGORIES).flatMap(([category, prompts]) => prompts.map((text) => ({ category, text }))),
  ...OTHER.map((text) => ({ category: "other", text })),
];

console.log(`Embedding ${entries.length} prompts…`);
const tokenizer = await AutoTokenizer.from_pretrained(MODEL, { revision: REVISION });
const textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL, { revision: REVISION, dtype: "fp32" });
// MobileCLIP's text encoder expects the full 77-token context, hence max_length padding.
const inputs = tokenizer(entries.map((e) => e.text), { padding: "max_length", truncation: true });
const { text_embeds } = await textModel(inputs);
const dim = text_embeds.dims[1];
const data = text_embeds.data as Float32Array;

const prompts: PromptsFile["prompts"] = entries.map((e, i) => {
  const v = Array.from(data.subarray(i * dim, (i + 1) * dim));
  const norm = Math.hypot(...v);
  return { ...e, embedding: v.map((x) => Math.round((x / norm) * 1e6) / 1e6) };
});

const file: PromptsFile = { model: MODEL, revision: REVISION, variant, dim, prompts };
await writeFile(`${OUT}/prompts.json`, JSON.stringify(file));
console.log(`Wrote ${OUT}/prompts.json (${Object.keys(CATEGORIES).join(", ")} + ${OTHER.length} other prompts)`);
