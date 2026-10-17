import path from "node:path";
import sharp from "sharp";
import { CLASSES, type ClassName } from "./config.ts";
import { createSession, loadOrt, type SessionOptions } from "./ort.ts";

// One libvips thread per image: inputs are shrunk to a few hundred pixels, so
// extra threads only add overhead (parallelism comes from concurrent requests).
sharp.concurrency(1);
// Every upload is a different image: libvips' operation cache (up to 50 MB by
// default) would only hold memory, never serve a hit.
sharp.cache(false);

export const MODEL_NAME = "MobileNetV2";
const SIZE = 224;

export type Predictions = Record<ClassName, number>;

export interface Classifier {
  name: string;
  classify(buffer: Buffer): Promise<Predictions>;
}

export class ImageDecodeError extends Error {
  constructor(cause?: unknown) {
    super("Unsupported or corrupt image", { cause });
  }
}

// Decodes any format sharp understands (JPEG, PNG, WebP, AVIF, GIF, TIFF, SVG...),
// applies EXIF rotation, flattens transparency onto white and resizes to the
// model's input size, the same stretch nsfwjs would apply itself.
export async function preprocess(buffer: Buffer, size: number) {
  try {
    const { data, info } = await sharp(buffer, { failOn: "error", limitInputPixels: 100_000_000 })
      .rotate()
      .flatten({ background: "#ffffff" })
      .resize(size, size, { fit: "fill" })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    return { data, width: info.width, height: info.height };
  } catch (err) {
    throw new ImageDecodeError(err);
  }
}

/**
 * nsfwjs's MobileNetV2, converted to ONNX at build time (scripts/prepare-nsfw.ts).
 * Same weights and preprocessing as nsfwjs: scores match it to ~1e-6.
 * @param opts.sessionOptions  extra ONNX Runtime session options
 */
export async function createClassifier({
  modelDir = "models/nsfw",
  sessionOptions = {},
}: { modelDir?: string; sessionOptions?: SessionOptions } = {}) {
  const { Tensor } = await loadOrt();
  const session = await createSession(path.join(modelDir, "mobilenet_v2.onnx"), sessionOptions);

  async function classify(buffer: Buffer): Promise<Predictions> {
    const { data } = await preprocess(buffer, SIZE);
    const input = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) input[i] = data[i] / 255; // NHWC, 0..1, as nsfwjs
    const { probabilities } = await session.run({ input: new Tensor("float32", input, [1, SIZE, SIZE, 3]) });
    const scores = probabilities.data as Float32Array;
    return Object.fromEntries(CLASSES.map((name, i) => [name, scores[i]])) as Predictions;
  }

  // Run one inference on a blank image so the first real request isn't slow.
  await classify(await sharp({ create: { width: SIZE, height: SIZE, channels: 3, background: "#808080" } }).png().toBuffer());

  return { name: MODEL_NAME, classify, dispose: () => session.release() };
}
