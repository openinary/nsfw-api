import { describe, expect, it } from "vitest";
import { createApp } from "../src/app.js";
import { ImageDecodeError } from "../src/classifier.js";
import { makeImage, multipart, testConfig } from "./helpers.js";

// HTTP behaviour with a stub classifier: fast, no model involved.
const scores = { Drawing: 0.05, Hentai: 0.05, Neutral: 0.6, Porn: 0.2, Sexy: 0.1 };

function stub(classify = async () => scores) {
  return { name: "Stub", classify };
}

function appWith(config = {}, classifier = stub(), clip = null) {
  return createApp({
    config: testConfig(config),
    getClassifier: () => classifier,
    getClip: () => clip,
  });
}

const clipStub = (scores) => ({
  classify: async () =>
    Object.fromEntries(Object.entries(scores).map(([k, score]) => [k, { score, match: `a photo of ${k}` }])),
});

const post = (app, body, { headers = {}, query = "" } = {}) =>
  app.request(`/classify${query}`, { method: "POST", body, headers });

describe("GET /health", () => {
  it("returns 503 while the model loads", async () => {
    const app = createApp({ config: testConfig(), getClassifier: () => null });
    const res = await app.request("/health");
    expect(res.status).toBe(503);
    expect(await res.json()).toEqual({ status: "loading" });
  });

  it("returns 200 with model info once ready", async () => {
    const res = await appWith().request("/health");
    expect(res.status).toBe(200);
    expect(await res.json()).toMatchObject({
      status: "ok",
      model: "Stub",
      authRequired: false,
      threshold: 0.7,
      nsfwClasses: ["Porn", "Hentai", "Sexy"],
      maxFileSizeMb: 10,
    });
  });
});

describe("GET /", () => {
  it("is the healthcheck, identical to /health", async () => {
    const app = appWith();
    const root = await app.request("/");
    expect(root.status).toBe(200);
    expect(root.headers.get("content-type")).toContain("application/json");
    expect(await root.json()).toEqual(await (await app.request("/health")).json());
  });

  it("returns 503 while the model loads", async () => {
    const app = createApp({ config: testConfig(), getClassifier: () => null });
    expect((await app.request("/")).status).toBe(503);
  });
});

describe("GET /playground", () => {
  it("serves the playground by default", async () => {
    const res = await appWith().request("/playground");
    expect(res.headers.get("content-type")).toContain("text/html");
    expect(await res.text()).toContain("NSFW Detection API");
  });

  it("is a 404 when ENABLE_PLAYGROUND=false", async () => {
    expect((await appWith({ enablePlayground: false }).request("/playground")).status).toBe(404);
  });
});

describe("POST /classify", () => {
  it("accepts multipart uploads and computes the verdict", async () => {
    const res = await post(appWith(), multipart(await makeImage()));
    expect(res.status).toBe(200);
    const body = await res.json();
    // default NSFW classes are Porn + Hentai + Sexy = 0.35, below 0.7
    expect(body).toMatchObject({ nsfw: false, score: 0.35, threshold: 0.7, model: "Stub" });
    expect(body.top).toEqual({ className: "Neutral", probability: 0.6 });
    expect(Object.keys(body.predictions).sort()).toEqual(["Drawing", "Hentai", "Neutral", "Porn", "Sexy"]);
  });

  it("accepts a raw image body", async () => {
    const res = await post(appWith(), await makeImage("png"), { headers: { "content-type": "image/png" } });
    expect(res.status).toBe(200);
  });

  it("uses any file field when 'image' is absent", async () => {
    const res = await post(appWith(), multipart(await makeImage(), { field: "file" }));
    expect(res.status).toBe(200);
  });

  it("honours ?threshold and NSFW_CLASSES", async () => {
    const app = appWith({ nsfwClasses: ["Porn"] });
    expect(await (await post(app, multipart(await makeImage()), { query: "?threshold=0.1" })).json()).toMatchObject({
      nsfw: true,
      score: 0.2,
      threshold: 0.1,
    });
  });

  it.each(["-1", "2", "abc"])("rejects threshold=%s", async (t) => {
    const res = await post(appWith(), multipart(await makeImage()), { query: `?threshold=${t}` });
    expect(res.status).toBe(400);
  });

  it("returns 400 without an image", async () => {
    expect((await post(appWith(), new FormData())).status).toBe(400);
    expect((await post(appWith(), "")).status).toBe(400);
  });

  it("returns 413 above MAX_FILE_SIZE", async () => {
    const res = await post(appWith({ maxFileSize: 1024 }), Buffer.alloc(4096), {
      headers: { "content-type": "image/jpeg" },
    });
    expect(res.status).toBe(413);
  });

  it("returns 415 when the image cannot be decoded", async () => {
    const app = appWith({}, stub(async () => { throw new ImageDecodeError(); }));
    const res = await post(app, multipart(Buffer.from("not an image")));
    expect(res.status).toBe(415);
  });

  it("returns 503 while the model loads", async () => {
    const app = createApp({ config: testConfig(), getClassifier: () => null });
    expect((await post(app, multipart(await makeImage()))).status).toBe(503);
  });

  it("waits for the models instead of failing when they are about to be ready", async () => {
    let classifier = null;
    let markReady;
    const ready = new Promise((r) => (markReady = r));
    const app = createApp({ config: testConfig(), getClassifier: () => classifier, whenReady: () => ready });
    const pending = post(app, multipart(await makeImage()));
    setTimeout(() => {
      classifier = stub();
      markReady();
    }, 50);
    expect((await pending).status).toBe(200);
  });

  it("gives up with 503 when the models take longer than readyTimeoutMs", async () => {
    const app = createApp({
      config: testConfig(),
      getClassifier: () => null,
      whenReady: () => new Promise(() => {}),
      readyTimeoutMs: 50,
    });
    expect((await post(app, multipart(await makeImage()))).status).toBe(503);
  });

  it("returns 429 when the queue is full", async () => {
    let release;
    const gate = new Promise((r) => (release = r));
    const app = appWith({ maxConcurrency: 1, maxQueue: 0 }, stub(async () => (await gate, scores)));
    const img = await makeImage();
    const first = post(app, multipart(img));
    await new Promise((r) => setTimeout(r, 20));
    const second = await post(app, multipart(img));
    expect(second.status).toBe(429);
    expect(second.headers.get("retry-after")).toBe("1");
    release();
    expect((await first).status).toBe(200);
  });

  it("hides internal errors", async () => {
    const app = appWith({}, stub(async () => { throw new Error("secret detail"); }));
    const res = await post(app, multipart(await makeImage()));
    expect(res.status).toBe(500);
    expect(await res.text()).not.toContain("secret");
  });
});

describe("API key", () => {
  const app = appWith({ apiKey: "s3cret" });

  it("rejects missing or wrong keys", async () => {
    const img = await makeImage();
    expect((await post(app, multipart(img))).status).toBe(401);
    expect((await post(app, multipart(img), { headers: { authorization: "Bearer nope" } })).status).toBe(401);
  });

  it("accepts Bearer and x-api-key", async () => {
    const img = await makeImage();
    expect((await post(app, multipart(img), { headers: { authorization: "Bearer s3cret" } })).status).toBe(200);
    expect((await post(app, multipart(img), { headers: { "x-api-key": "s3cret" } })).status).toBe(200);
  });

  it("keeps /health public", async () => {
    expect((await app.request("/health")).status).toBe(200);
  });
});

describe("CLIP categories", () => {
  const nsfwLow = stub(async () => ({ Drawing: 0, Hentai: 0, Neutral: 1, Porn: 0, Sexy: 0 }));

  it("adds categories, flags and safe to the response", async () => {
    const app = appWith({ clipThreshold: 0.5 }, nsfwLow, clipStub({ weapon: 0.91, gore: 0.02 }));
    const body = await (await post(app, multipart(await makeImage()))).json();
    expect(body).toMatchObject({
      safe: false,
      flags: ["weapon"],
      nsfw: false,
      categories: {
        weapon: { score: 0.91, flagged: true, match: "a photo of weapon" },
        gore: { score: 0.02, flagged: false },
      },
    });
  });

  it("is safe when nothing crosses a threshold", async () => {
    const app = appWith({}, nsfwLow, clipStub({ weapon: 0.1 }));
    expect(await (await post(app, multipart(await makeImage()))).json()).toMatchObject({ safe: true, flags: [] });
  });

  it("lists nsfw first when both trigger", async () => {
    const app = appWith({}, stub(async () => ({ Drawing: 0, Hentai: 0, Neutral: 0, Porn: 1, Sexy: 0 })), clipStub({ gore: 0.9 }));
    expect((await (await post(app, multipart(await makeImage()))).json()).flags).toEqual(["nsfw", "gore"]);
  });

  it("omits categories when CLIP is disabled", async () => {
    const body = await (await post(appWith(), multipart(await makeImage()))).json();
    expect(body).not.toHaveProperty("categories");
    expect(body).toMatchObject({ safe: true, flags: [] });
  });

  it("describes CLIP in /health", async () => {
    const on = await (await appWith({ clipCategories: ["weapon"], clipThreshold: 0.6 }, stub(), clipStub({})).request("/health")).json();
    expect(on.clip).toEqual({ categories: ["weapon"], threshold: 0.6 });
    expect((await (await appWith().request("/health")).json()).clip).toBeNull();
  });
});
