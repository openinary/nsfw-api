// Smoke test against a running instance (local, Docker or Railway).
//   node scripts/smoke.js https://your-app.up.railway.app
//   API_KEY=... node scripts/smoke.js http://localhost:8080
import sharp from "sharp";

const base = (process.argv[2] || process.env.BASE_URL || "http://localhost:8080").replace(/\/$/, "");
const apiKey = process.env.API_KEY || "";
const auth = apiKey ? { authorization: `Bearer ${apiKey}` } : {};

let failed = 0;
async function check(name, fn) {
  try {
    const detail = await fn();
    console.log(`✓ ${name}${detail ? `  ${detail}` : ""}`);
  } catch (err) {
    failed++;
    console.log(`✗ ${name}  ${err.message}`);
  }
}
function expectStatus(res, status) {
  if (res.status !== status) throw new Error(`expected ${status}, got ${res.status}`);
}

const image = await sharp({ create: { width: 800, height: 600, channels: 3, background: "#4a90d9" } }).jpeg().toBuffer();
const form = (buf) => {
  const f = new FormData();
  f.append("image", new Blob([buf], { type: "image/jpeg" }), "smoke.jpg");
  return f;
};

console.log(`Smoke testing ${base}\n`);

let health;
await check("GET /health is ready", async () => {
  const res = await fetch(`${base}/health`);
  expectStatus(res, 200);
  health = await res.json();
  return `model=${health.model} auth=${health.authRequired}`;
});

await check("POST /classify (multipart)", async () => {
  const res = await fetch(`${base}/classify`, { method: "POST", body: form(image), headers: auth });
  expectStatus(res, 200);
  const body = await res.json();
  return `nsfw=${body.nsfw} score=${body.score} top=${body.top.className} ${body.timeMs}ms`;
});

await check("POST /classify (raw body)", async () => {
  const res = await fetch(`${base}/classify`, {
    method: "POST",
    body: image,
    headers: { ...auth, "content-type": "image/jpeg" },
  });
  expectStatus(res, 200);
});

await check("corrupt image → 415", async () => {
  const res = await fetch(`${base}/classify`, { method: "POST", body: form(Buffer.from("nope")), headers: auth });
  expectStatus(res, 415);
});

if (health?.authRequired) {
  await check("missing API key → 401", async () => {
    const res = await fetch(`${base}/classify`, { method: "POST", body: form(image) });
    expectStatus(res, 401);
  });
}

await check("10 parallel requests", async () => {
  const started = Date.now();
  const results = await Promise.all(
    Array.from({ length: 10 }, () => fetch(`${base}/classify`, { method: "POST", body: form(image), headers: auth })),
  );
  const codes = results.map((r) => r.status);
  if (codes.some((c) => c !== 200)) throw new Error(`statuses: ${codes.join(",")}`);
  return `${Date.now() - started}ms total`;
});

console.log(failed ? `\n${failed} check(s) failed` : "\nAll checks passed");
process.exit(failed ? 1 : 0);
