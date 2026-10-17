import { beforeAll, describe, expect, it } from "vitest";
import { createApp } from "../src/app.ts";
import { createClassifier, ImageDecodeError } from "../src/classifier.ts";
import { CLASSES } from "../src/config.ts";
import { makeImage, multipart, testConfig, type ImageFormat } from "./helpers.ts";

// End-to-end through nsfwjs's MobileNetV2 converted to ONNX (needs `npm run prepare-models`).
let classifier: Awaited<ReturnType<typeof createClassifier>>;
let app: ReturnType<typeof createApp>;

beforeAll(async () => {
  classifier = await createClassifier();
  app = createApp({ config: testConfig(), getClassifier: () => classifier });
}, 60_000);

const sum = (o: Record<string, number>) => Object.values(o).reduce((a, b) => a + b, 0);

describe(`real model (MobileNetV2)`, () => {
  it.each<[ImageFormat, object]>([
    ["jpeg", {}],
    ["png", {}],
    ["png", { alpha: true }],
    ["webp", {}],
    ["avif", {}],
    ["gif", {}],
    ["tiff", {}],
    ["jpeg", { width: 4000, height: 3000 }],
    ["png", { width: 16, height: 16 }],
  ])("classifies %s %o", async (format, opts) => {
    const scores = await classifier.classify(await makeImage(format, opts));
    expect(Object.keys(scores).sort()).toEqual(CLASSES);
    expect(sum(scores)).toBeCloseTo(1, 3);
  });

  it("is deterministic", async () => {
    const img = await makeImage();
    expect(await classifier.classify(img)).toEqual(await classifier.classify(img));
  });

  // Reference scores from nsfwjs 4.4.0 itself (TensorFlow.js) on the same images:
  // the ONNX conversion must not move them.
  it.each<[ImageFormat, object, Record<string, number>]>([
    ["jpeg", {}, { Drawing: 0.903305, Hentai: 0.017312, Neutral: 0.075418, Porn: 0.003189, Sexy: 0.000777 }],
    ["png", { alpha: true }, { Drawing: 0.784016, Hentai: 0.012766, Neutral: 0.198689, Porn: 0.00366, Sexy: 0.000868 }],
  ])("matches nsfwjs's own scores on %s %o", async (format, opts, expected) => {
    const scores = await classifier.classify(await makeImage(format, opts));
    for (const name of CLASSES) expect(scores[name]).toBeCloseTo(expected[name], 5);
  });

  it("rejects corrupt data with ImageDecodeError", async () => {
    await expect(classifier.classify(Buffer.from("definitely not an image"))).rejects.toBeInstanceOf(ImageDecodeError);
    const truncated = (await makeImage("jpeg")).subarray(0, 200);
    await expect(classifier.classify(truncated)).rejects.toBeInstanceOf(ImageDecodeError);
  });

  it("serves a full HTTP round trip", async () => {
    const res = await app.request("/classify", { method: "POST", body: multipart(await makeImage("webp")) });
    expect(res.status).toBe(200);
    const body: any = await res.json();
    expect(body.model).toBe("MobileNetV2");
    expect(typeof body.nsfw).toBe("boolean");
    expect(body.score).toBeGreaterThanOrEqual(0);
    expect(body.score).toBeLessThanOrEqual(1);
    console.log(`  timeMs=${body.timeMs} top=${body.top.className}`);
  });

  it("returns 415 over HTTP for a corrupt upload", async () => {
    const res = await app.request("/classify", { method: "POST", body: multipart(Buffer.from("nope")) });
    expect(res.status).toBe(415);
  });

  it("threshold 0 flags and threshold 1 clears the same image", async () => {
    const img = await makeImage();
    const at = async (t: number) => {
      const body: any = await (await app.request(`/classify?threshold=${t}`, { method: "POST", body: multipart(img) })).json();
      return body.nsfw;
    };
    expect(await at(0)).toBe(true);
    expect(await at(1)).toBe(false);
  });
});
