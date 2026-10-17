import { describe, expect, it } from "vitest";
import { createLimiter, QueueFullError } from "../src/limiter.js";

const tick = () => new Promise((r) => setTimeout(r, 5));

describe("createLimiter", () => {
  it("never runs more than `concurrency` jobs at once", async () => {
    const limiter = createLimiter({ concurrency: 2, maxQueue: 10 });
    let running = 0;
    let peak = 0;
    const job = async () => {
      running++;
      peak = Math.max(peak, running);
      await tick();
      running--;
    };
    await Promise.all(Array.from({ length: 8 }, () => limiter.run(job)));
    expect(peak).toBe(2);
    expect(limiter.active).toBe(0);
    expect(limiter.queued).toBe(0);
  });

  it("rejects once the queue is full", async () => {
    const limiter = createLimiter({ concurrency: 1, maxQueue: 1 });
    const slow = () => limiter.run(tick);
    const a = slow();
    const b = slow();
    await expect(slow()).rejects.toBeInstanceOf(QueueFullError);
    await Promise.all([a, b]);
  });

  it("frees the slot when a job throws", async () => {
    const limiter = createLimiter({ concurrency: 1, maxQueue: 0 });
    await expect(limiter.run(async () => { throw new Error("boom"); })).rejects.toThrow("boom");
    await expect(limiter.run(async () => 42)).resolves.toBe(42);
  });
});
