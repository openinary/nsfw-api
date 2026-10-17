import { readFileSync } from "node:fs";
import { timingSafeEqual } from "node:crypto";
import { Hono, type Context } from "hono";
import { bodyLimit } from "hono/body-limit";
import { cors } from "hono/cors";
import { ImageDecodeError, type Classifier, type Predictions } from "./classifier.ts";
import type { CategoryScore, ClipClassifier } from "./clip.ts";
import type { Config } from "./config.ts";
import { createLimiter, QueueFullError } from "./limiter.ts";

const playgroundHtml = readFileSync(new URL("./playground.html", import.meta.url), "utf8");

function safeEqual(a: string, b: string) {
  const x = Buffer.from(a);
  const y = Buffer.from(b);
  return x.length === y.length && timingSafeEqual(x, y);
}

export interface AppOptions {
  config: Config;
  /** Returns null while the model is still loading. */
  getClassifier: () => Classifier | null;
  /** The optional CLIP categories classifier; null when disabled. */
  getClip?: () => ClipClassifier | null;
  /**
   * Resolves once the models are loaded; /classify waits on it (up to readyTimeoutMs)
   * instead of failing, so a request that wakes a sleeping service still gets an answer.
   */
  whenReady?: () => Promise<unknown>;
  readyTimeoutMs?: number;
}

export function createApp({ config, getClassifier, getClip = () => null, whenReady, readyTimeoutMs = 60_000 }: AppOptions) {
  const app = new Hono();
  const limiter = createLimiter({ concurrency: config.maxConcurrency, maxQueue: config.maxQueue });

  if (config.corsOrigin) {
    app.use("*", cors({ origin: config.corsOrigin.split(",").map((s) => s.trim()) }));
  }

  app.onError((err, c) => {
    console.error(err);
    return c.json({ error: "Internal server error" }, 500);
  });

  app.notFound((c) => c.json({ error: "Not found" }, 404));

  if (config.enablePlayground) {
    app.get("/playground", (c) => c.html(playgroundHtml));
  }

  // "/" is the healthcheck; "/health" stays as an explicit alias (railway.json uses it).
  const health = (c: Context) => {
    const classifier = getClassifier();
    if (!classifier) return c.json({ status: "loading" }, 503);
    return c.json({
      status: "ok",
      model: classifier.name,
      authRequired: Boolean(config.apiKey),
      threshold: config.threshold,
      nsfwClasses: config.nsfwClasses,
      clip: getClip() ? { categories: config.clipCategories, threshold: config.clipThreshold } : null,
      maxFileSizeMb: config.maxFileSize / 1024 / 1024,
      inFlight: limiter.active,
      queued: limiter.queued,
    });
  };
  app.get("/", health);
  app.get("/health", health);

  app.post(
    "/classify",
    async (c, next) => {
      if (!config.apiKey) return next();
      const header = c.req.header("authorization") || "";
      const token = header.startsWith("Bearer ") ? header.slice(7) : c.req.header("x-api-key") || "";
      if (!safeEqual(token, config.apiKey)) return c.json({ error: "Invalid or missing API key" }, 401);
      return next();
    },
    bodyLimit({
      maxSize: config.maxFileSize,
      onError: (c) => c.json({ error: `Image larger than ${config.maxFileSize / 1024 / 1024} MB` }, 413),
    }),
    async (c) => {
      if (!getClassifier() && whenReady) {
        let timer: NodeJS.Timeout | undefined;
        const timeout = new Promise((resolve) => (timer = setTimeout(resolve, readyTimeoutMs)));
        await Promise.race([whenReady(), timeout]).finally(() => clearTimeout(timer));
      }
      const classifier = getClassifier();
      if (!classifier) return c.json({ error: "Model is still loading, retry in a few seconds" }, 503);

      let threshold = config.threshold;
      const t = c.req.query("threshold");
      if (t !== undefined) {
        threshold = Number(t);
        if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
          return c.json({ error: "threshold must be a number between 0 and 1" }, 400);
        }
      }

      const image = await readImage(c);
      if (!image) {
        return c.json(
          { error: 'Send an image as multipart/form-data (field "image") or as a raw image/* body' },
          400,
        );
      }

      const started = performance.now();
      const clip = getClip();
      let predictions: Predictions, categories: Record<string, CategoryScore> | null;
      try {
        // Both models run on ONNX Runtime's native threads: run them side by side.
        [predictions, categories] = await limiter.run(() =>
          Promise.all([classifier.classify(image), clip ? clip.classify(image) : null]),
        );
      } catch (err) {
        if (err instanceof ImageDecodeError) return c.json({ error: err.message }, 415);
        if (err instanceof QueueFullError) {
          c.header("Retry-After", "1");
          return c.json({ error: err.message }, 429);
        }
        throw err;
      }

      const score = config.nsfwClasses.reduce((sum, name) => sum + predictions[name], 0);
      const [topClass, topProbability] = Object.entries(predictions).sort((a, b) => b[1] - a[1])[0];
      const nsfw = score > threshold;
      const flags = nsfw ? ["nsfw"] : [];
      let flagged: Record<string, CategoryScore & { flagged: boolean }> | undefined;
      if (categories) {
        flagged = {};
        for (const [name, { score, match }] of Object.entries(categories)) {
          flagged[name] = { score: round(score), match, flagged: score > config.clipThreshold };
          if (flagged[name].flagged) flags.push(name);
        }
      }

      return c.json({
        safe: flags.length === 0,
        flags,
        nsfw,
        score: round(score),
        threshold,
        top: { className: topClass, probability: round(topProbability) },
        predictions: Object.fromEntries(Object.entries(predictions).map(([k, v]) => [k, round(v)])),
        ...(flagged && { categories: flagged }),
        model: classifier.name,
        timeMs: Math.round(performance.now() - started),
      });
    },
  );

  return app;
}

const round = (n: number) => Math.round(n * 10_000) / 10_000;

async function readImage(c: Context): Promise<Buffer | null> {
  const type = c.req.header("content-type") || "";
  if (type.startsWith("multipart/form-data")) {
    const form = await c.req.formData();
    const file = form.get("image") ?? [...form.values()].find((v) => typeof v === "object");
    if (!file || typeof file === "string") return null;
    return Buffer.from(await file.arrayBuffer());
  }
  const body = Buffer.from(await c.req.arrayBuffer());
  return body.length ? body : null;
}
