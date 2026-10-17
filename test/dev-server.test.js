import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { afterAll, expect, it } from "vitest";

// `npm run dev` runs src/ with plain Node, whose ES module resolution is
// stricter than Vitest's or esbuild's (e.g. no extensionless deep imports).
// Starting the real process catches what the in-process tests can't.
const freePort = () =>
  new Promise((resolve) => {
    const srv = createServer().listen(0, () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });

let proc;
afterAll(() => proc?.kill());

it("starts from source with plain Node", async () => {
  const port = await freePort();
  let output = "";
  proc = spawn(process.execPath, ["src/server.js"], {
    env: { ...process.env, PORT: String(port), API_KEY: "" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  proc.stdout.on("data", (d) => (output += d));
  proc.stderr.on("data", (d) => (output += d));

  let health;
  for (let i = 0; i < 150 && proc.exitCode === null; i++) {
    try {
      const res = await fetch(`http://localhost:${port}/health`);
      if (res.status === 200) {
        health = await res.json();
        break;
      }
    } catch {}
    await new Promise((r) => setTimeout(r, 100));
  }
  expect(health, output).toMatchObject({ status: "ok", model: "MobileNetV2" });
}, 30_000);
