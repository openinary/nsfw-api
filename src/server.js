import { serve } from "@hono/node-server";
import { createApp } from "./app.js";
import { createClassifier } from "./classifier.js";
import { createClipClassifier } from "./clip.js";
import { loadConfig } from "./config.js";

const config = loadConfig();

let classifier = null;
let clip = null;
let markReady;
const ready = new Promise((resolve) => (markReady = resolve));

const app = createApp({ config, getClassifier: () => classifier, getClip: () => clip, whenReady: () => ready });

// Listen right away so the platform sees the port open; /health answers 503
// until the models are ready, which is what Railway's healthcheck waits for.
const server = serve({ fetch: app.fetch, port: config.port, hostname: "::" }, (info) => {
  console.log(`[server] listening on :${info.port}`);
});

try {
  const started = Date.now();
  const [nsfw, categories] = await Promise.all([
    createClassifier(),
    config.clipCategories.length
      ? createClipClassifier({ categories: config.clipCategories, threads: config.clipThreads })
      : null,
  ]);
  clip = categories;
  classifier = nsfw; // /health turns green only once every model is loaded
  markReady();
  console.log(
    `[models] ${nsfw.name}${clip ? ` + CLIP ${clip.variant} (${config.clipCategories.join(", ")})` : ""} ready in ${Date.now() - started} ms`,
  );
  if (!config.apiKey) console.warn("[auth] API_KEY is not set: /classify is open to anyone");
} catch (err) {
  console.error("[models] failed to load:", err);
  process.exit(1);
}

function shutdown(signal) {
  console.log(`[server] ${signal} received, closing`);
  server.close(() => {
    classifier?.dispose();
    clip?.dispose();
    process.exit(0);
  });
  setTimeout(() => process.exit(0), 10_000).unref();
}
process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
