import { existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import sharp from "sharp";
import { beforeAll, describe, expect, it } from "vitest";
import { ImageDecodeError } from "../src/classifier.js";
import { createClipClassifier } from "../src/clip.js";
import { makeImage } from "./helpers.js";

// Needs the build-time artefacts: run `npm run prepare-models` once.
const ready = existsSync("models/clip/prompts.json") && existsSync("models/clip/vision.onnx");

const svg = (body) =>
  sharp(Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480">${body}</svg>`)).jpeg().toBuffer();

describe.skipIf(!ready)("CLIP categories (real model)", () => {
  let clip;
  beforeAll(async () => {
    clip = await createClipClassifier({ categories: ["weapon", "gore", "drugs"] });
  }, 60_000);

  it("scores every enabled category between 0 and 1", async () => {
    const result = await clip.classify(await makeImage("jpeg"));
    expect(Object.keys(result)).toEqual(["weapon", "gore", "drugs"]);
    const total = Object.values(result).reduce((s, c) => s + c.score, 0);
    for (const c of Object.values(result)) {
      expect(c.score).toBeGreaterThanOrEqual(0);
      expect(c.score).toBeLessThanOrEqual(1);
      expect(typeof c.match).toBe("string");
    }
    // categories share the probability mass with the other prompts
    expect(total).toBeLessThan(1);
  });

  it("keeps an obviously harmless picture low on every category", async () => {
    const landscape = await svg(
      '<rect width="640" height="480" fill="#8ec5ff"/><rect y="300" width="640" height="180" fill="#3a7d2c"/><circle cx="520" cy="90" r="50" fill="#ffd23f"/>',
    );
    for (const { score } of Object.values(await clip.classify(landscape))) expect(score).toBeLessThan(0.2);
  });

  it("is deterministic", async () => {
    const img = await makeImage("png");
    expect(await clip.classify(img)).toEqual(await clip.classify(img));
  });

  it("handles transparency, tiny and huge images", async () => {
    for (const img of [
      await makeImage("png", { alpha: true }),
      await makeImage("png", { width: 16, height: 16 }),
      await makeImage("jpeg", { width: 4000, height: 3000 }),
    ]) {
      await expect(clip.classify(img)).resolves.toHaveProperty("weapon");
    }
  });

  it("only reports the categories it was asked for", async () => {
    const weaponsOnly = await createClipClassifier({ categories: ["weapon"] });
    expect(Object.keys(await weaponsOnly.classify(await makeImage()))).toEqual(["weapon"]);
  });

  it("refuses prompts embedded with another CLIP model", async () => {
    const dir = mkdtempSync(path.join(tmpdir(), "clip-"));
    writeFileSync(path.join(dir, "prompts.json"), JSON.stringify({ model: "Xenova/clip-vit-base-patch32", prompts: [], dim: 512 }));
    await expect(createClipClassifier({ categories: ["weapon"], modelDir: dir })).rejects.toThrow(/prepare-models/);
  });

  it("rejects unknown categories and corrupt images", async () => {
    await expect(createClipClassifier({ categories: ["tanks"] })).rejects.toThrow(/Unknown CLIP category "tanks"/);
    await expect(clip.classify(Buffer.from("nope"))).rejects.toBeInstanceOf(ImageDecodeError);
  });
});
