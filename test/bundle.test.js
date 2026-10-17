import { execFileSync, spawn } from "node:child_process";
import { createServer } from "node:net";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { makeImage, multipart } from "./helpers.js";

// Builds dist/ exactly like the Dockerfile does, then runs it as a real process:
// catches bundling problems (missing globals, wrongly pruned modules) that
// source-level tests can't see.
const freePort = () =>
  new Promise((resolve) => {
    const srv = createServer().listen(0, () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });

async function start(env) {
  const port = await freePort();
  const proc = spawn(process.execPath, ["dist/server.js"], {
    env: { ...process.env, ...env, PORT: String(port) },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let output = "";
  proc.stdout.on("data", (d) => (output += d));
  proc.stderr.on("data", (d) => (output += d));
  const exited = new Promise((resolve) => proc.on("exit", resolve));
  return { proc, port, exited, output: () => output };
}

async function waitHealthy(port) {
  for (let i = 0; i < 100; i++) {
    try {
      const res = await fetch(`http://localhost:${port}/health`);
      if (res.status === 200) return res.json();
    } catch {}
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error("server never became healthy");
}

describe("production bundle", () => {
  let server;

  beforeAll(async () => {
    execFileSync(process.execPath, ["build.js"]);
    server = await start({ API_KEY: "" });
  }, 60_000);

  afterAll(() => server?.proc.kill());

  it("starts and loads the models", async () => {
    const health = await waitHealthy(server.port);
    expect(health).toMatchObject({ status: "ok", model: "MobileNetV2" });
    expect(server.output()).not.toMatch(/failed/i);
  });

  it("classifies an image quickly", async () => {
    const res = await fetch(`http://localhost:${server.port}/classify`, {
      method: "POST",
      body: multipart(await makeImage("webp")),
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.timeMs).toBeLessThan(500);
  });

  it("serves the playground and the root healthcheck", async () => {
    expect(await (await fetch(`http://localhost:${server.port}/`)).json()).toMatchObject({ status: "ok" });
    const res = await fetch(`http://localhost:${server.port}/playground`);
    expect(await res.text()).toContain("<html");
  });

  it("refuses to start with an invalid configuration", async () => {
    const other = await start({ NSFW_THRESHOLD: "2" });
    expect(await other.exited).toBe(1);
    expect(other.output()).toContain("NSFW_THRESHOLD");
  });
});
