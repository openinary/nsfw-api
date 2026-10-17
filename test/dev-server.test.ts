import { spawn, type ChildProcess } from "node:child_process";
import { createServer, type AddressInfo } from "node:net";
import { afterAll, expect, it } from "vitest";

// `npm run dev` runs src/ with plain Node, whose ES module resolution is
// stricter than Vitest's or esbuild's (e.g. no extensionless deep imports).
// Starting the real process catches what the in-process tests can't.
const freePort = () =>
  new Promise<number>((resolve) => {
    const srv = createServer().listen(0, () => {
      const { port } = srv.address() as AddressInfo;
      srv.close(() => resolve(port));
    });
  });

let proc: ChildProcess | undefined;
afterAll(() => proc?.kill());

it("starts from source with plain Node", async () => {
  const port = await freePort();
  let output = "";
  const child = (proc = spawn(process.execPath, ["src/server.ts"], {
    env: { ...process.env, PORT: String(port), API_KEY: "" },
    stdio: ["ignore", "pipe", "pipe"],
  }));
  child.stdout.on("data", (d) => (output += d));
  child.stderr.on("data", (d) => (output += d));

  let health;
  for (let i = 0; i < 150 && child.exitCode === null; i++) {
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
