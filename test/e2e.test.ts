import { spawn } from "node:child_process";
import { createServer, type AddressInfo } from "node:net";
import sharp from "sharp";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { makeImage, multipart, type ImageFormat } from "./helpers.ts";

// End-to-end through the release binary and the real models: needs
// `npm run prepare-models` and `cargo build --release` (`npm test` runs the latter).
// The HTTP edge cases that need a stub classifier are Rust tests (src/app.rs).
const CLASSES = ["Drawing", "Hentai", "Neutral", "Porn", "Sexy"];
// AVIF is a build feature (`cargo build --release --features avif`, what the Dockerfile does).
const avif = process.env.TEST_AVIF === "1";

const freePort = () =>
  new Promise<number>((resolve) => {
    const srv = createServer().listen(0, () => {
      const { port } = srv.address() as AddressInfo;
      srv.close(() => resolve(port));
    });
  });

async function start(env: Record<string, string> = {}) {
  const port = await freePort();
  const proc = spawn("target/release/nsfw-api", { env: { ...process.env, API_KEY: "", ...env, PORT: String(port) } });
  let output = "";
  proc.stdout.on("data", (d) => (output += d));
  proc.stderr.on("data", (d) => (output += d));
  const exited = new Promise<number | null>((resolve) => proc.on("exit", resolve));
  const url = `http://localhost:${port}`;
  const classify = async (image: Buffer, query = "") => {
    const res = await fetch(`${url}/classify${query}`, { method: "POST", body: multipart(image) });
    return { status: res.status, body: (await res.json()) as any };
  };
  const healthy = async () => {
    for (let i = 0; i < 100 && proc.exitCode === null; i++) {
      try {
        const res = await fetch(`${url}/health`);
        if (res.status === 200) return res.json() as any;
      } catch {}
      await new Promise((r) => setTimeout(r, 100));
    }
    throw new Error(`server never became healthy:\n${output}`);
  };
  return { proc, url, exited, output: () => output, classify, healthy };
}

const svg = (body: string) => Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480">${body}</svg>`);
const landscape = svg(
  '<rect width="640" height="480" fill="#8ec5ff"/><rect y="300" width="640" height="180" fill="#3a7d2c"/><circle cx="520" cy="90" r="50" fill="#ffd23f"/>',
);

describe("release binary with the real models", () => {
  let server: Awaited<ReturnType<typeof start>>;

  beforeAll(async () => {
    server = await start();
    await server.healthy();
  }, 60_000);

  afterAll(() => server?.proc.kill());

  it("starts and loads the models", async () => {
    expect(await server.healthy()).toMatchObject({
      status: "ok",
      model: "MobileNetV2Mid",
      clip: { categories: ["weapon", "gore", "drugs"], threshold: 0.5 },
    });
    expect(server.output()).toMatch(/\[models\] MobileNetV2Mid \+ CLIP \w+ \(weapon, gore, drugs\) ready/);
    expect(server.output()).not.toMatch(/failed/i);
  });

  it.each<[ImageFormat, { width?: number; height?: number; alpha?: boolean }]>([
    ["jpeg", {}],
    ["png", {}],
    ["png", { alpha: true }],
    ["webp", {}],
    ["gif", {}],
    ["tiff", {}],
    ["jpeg", { width: 4000, height: 3000 }],
    ["png", { width: 16, height: 16 }],
    ...(avif ? ([["avif", {}]] as [ImageFormat, {}][]) : []),
  ])("classifies %s %o", async (format, opts) => {
    // sharp's default TIFF is JPEG-compressed YCbCr, which the decoder doesn't read: use LZW.
    const image =
      format === "tiff" ? await sharp(await makeImage("png")).tiff({ compression: "lzw" }).toBuffer() : await makeImage(format, opts);
    const { status, body } = await server.classify(image);
    expect(status).toBe(200);
    expect(Object.keys(body.predictions)).toEqual(CLASSES);
    expect(Object.values<number>(body.predictions).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 3);
    expect(Object.keys(body.categories)).toEqual(["weapon", "gore", "drugs"]);
    for (const c of Object.values<any>(body.categories)) {
      expect(c.score).toBeGreaterThanOrEqual(0);
      expect(c.score).toBeLessThanOrEqual(1);
      expect(typeof c.match).toBe("string");
    }
  });

  // Reference scores from nsfwjs 4.4.0 itself (TensorFlow.js) on the same pixels.
  // 224×224 lossless images need no resize, so this checks the ONNX conversion
  // and the inference engine alone (the API rounds to 4 decimals).
  it.each<[object, Record<string, number>]>([
    [{}, { Drawing: 0.078739, Hentai: 0.027847, Neutral: 0.892327, Porn: 0.000425, Sexy: 0.000662 }],
    [{ alpha: true }, { Drawing: 0.065441, Hentai: 0.02602, Neutral: 0.906642, Porn: 0.000865, Sexy: 0.001033 }],
  ])("matches nsfwjs's own scores on a 224×224 png %o", async (opts, expected) => {
    const { body } = await server.classify(await makeImage("png", { width: 224, height: 224, ...opts }));
    for (const name of CLASSES) expect(body.predictions[name]).toBeCloseTo(expected[name], 4);
  });

  it("is deterministic", async () => {
    const img = await makeImage();
    const strip = ({ timeMs, ...rest }: any) => rest;
    expect(strip((await server.classify(img)).body)).toEqual(strip((await server.classify(img)).body));
  });

  it("applies EXIF rotation", async () => {
    const upright = await makeImage("png", { width: 480, height: 640 });
    const sideways = await sharp(upright).rotate(-90).withMetadata({ orientation: 6 }).png().toBuffer();
    const [a, b] = [await server.classify(upright), await server.classify(sideways)];
    for (const name of CLASSES) expect(b.body.predictions[name]).toBeCloseTo(a.body.predictions[name], 3);
  });

  it("renders SVG uploads", async () => {
    const [vector, raster] = [await server.classify(landscape), await server.classify(await sharp(landscape).png().toBuffer())];
    expect(vector.status).toBe(200);
    for (const name of CLASSES) expect(vector.body.predictions[name]).toBeCloseTo(raster.body.predictions[name], 2);
  });

  it("keeps an obviously harmless picture low on every category", async () => {
    const { body } = await server.classify(await sharp(landscape).jpeg().toBuffer());
    expect(body).toMatchObject({ safe: true, flags: [], nsfw: false });
    for (const { score } of Object.values<any>(body.categories)) expect(score).toBeLessThan(0.2);
  });

  it("returns 415 for corrupt uploads", async () => {
    expect((await server.classify(Buffer.from("definitely not an image"))).status).toBe(415);
    expect((await server.classify((await makeImage("jpeg")).subarray(0, 200))).status).toBe(415);
  });

  it("threshold 0 flags and threshold 1 clears the same image", async () => {
    const img = await makeImage();
    expect((await server.classify(img, "?threshold=0")).body.nsfw).toBe(true);
    expect((await server.classify(img, "?threshold=1")).body.nsfw).toBe(false);
  });

  it("classifies an image quickly", async () => {
    const { body } = await server.classify(await makeImage("webp"));
    expect(body.timeMs).toBeLessThan(500);
    console.log(`  timeMs=${body.timeMs} top=${body.top.className}`);
  });

  it("serves the playground and the root healthcheck", async () => {
    expect(await (await fetch(`${server.url}/`)).json()).toMatchObject({ status: "ok" });
    expect(await (await fetch(`${server.url}/playground`)).text()).toContain("<html");
  });

  it("stops on SIGTERM", async () => {
    const other = await start({ CLIP_CATEGORIES: "none" });
    await other.healthy();
    other.proc.kill("SIGTERM");
    expect(await other.exited).toBe(0);
  });
});

describe("configuration", () => {
  it("only reports the categories it was asked for", async () => {
    const server = await start({ CLIP_CATEGORIES: "weapon" });
    await server.healthy();
    expect(Object.keys((await server.classify(await makeImage())).body.categories)).toEqual(["weapon"]);
    server.proc.kill();
  });

  it("runs without CLIP", async () => {
    const server = await start({ CLIP_CATEGORIES: "none" });
    expect(await server.healthy()).toMatchObject({ clip: null });
    expect((await server.classify(await makeImage())).body).not.toHaveProperty("categories");
    server.proc.kill();
  });

  it.each([
    [{ NSFW_THRESHOLD: "2" }, "NSFW_THRESHOLD"],
    [{ CLIP_CATEGORIES: "tanks" }, 'Unknown CLIP category "tanks"'],
  ])("refuses to start with %o", async (env, message) => {
    const server = await start(env);
    expect(await server.exited).toBe(1);
    expect(server.output()).toContain(message);
  });
});
