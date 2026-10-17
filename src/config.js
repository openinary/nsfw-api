export const CLASSES = ["Drawing", "Hentai", "Neutral", "Porn", "Sexy"];

function number(env, name, fallback, { min = -Infinity, max = Infinity } = {}) {
  const raw = env[name];
  if (raw === undefined || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < min || value > max) {
    throw new Error(`${name} must be a number between ${min} and ${max} (got "${raw}")`);
  }
  return value;
}

export function loadConfig(env = process.env) {
  const nsfwClasses = (env.NSFW_CLASSES || "Porn,Hentai,Sexy")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  for (const name of nsfwClasses) {
    if (!CLASSES.includes(name)) {
      throw new Error(`NSFW_CLASSES contains unknown class "${name}" (allowed: ${CLASSES.join(", ")})`);
    }
  }

  const clipCategories = (env.CLIP_CATEGORIES ?? "weapon,gore,drugs")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s && s !== "none");

  return {
    port: number(env, "PORT", 8080, { min: 1, max: 65535 }),
    apiKey: env.API_KEY || "",
    threshold: number(env, "NSFW_THRESHOLD", 0.7, { min: 0, max: 1 }),
    nsfwClasses,
    maxFileSize: number(env, "MAX_FILE_SIZE_MB", 10, { min: 0.1, max: 100 }) * 1024 * 1024,
    maxConcurrency: number(env, "MAX_CONCURRENCY", 2, { min: 1, max: 64 }),
    maxQueue: number(env, "MAX_QUEUE", 32, { min: 0, max: 10_000 }),
    clipCategories,
    clipThreshold: number(env, "CLIP_THRESHOLD", 0.5, { min: 0, max: 1 }),
    clipThreads: number(env, "CLIP_THREADS", 1, { min: 1, max: 64 }),
    enablePlayground: env.ENABLE_PLAYGROUND !== "false",
    corsOrigin: env.CORS_ORIGIN || "",
  };
}
