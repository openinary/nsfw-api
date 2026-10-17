export class QueueFullError extends Error {
  constructor() {
    super("Server busy, retry later");
  }
}

export interface Limiter {
  readonly active: number;
  readonly queued: number;
  run<T>(fn: () => Promise<T>): Promise<T>;
}

// Caps how many inferences run at once and how many may wait behind them.
export function createLimiter({ concurrency, maxQueue }: { concurrency: number; maxQueue: number }): Limiter {
  let active = 0;
  const waiting: (() => void)[] = [];

  const next = () => {
    active--;
    const resolve = waiting.shift();
    if (resolve) {
      active++;
      resolve();
    }
  };

  return {
    get active() {
      return active;
    },
    get queued() {
      return waiting.length;
    },
    async run(fn) {
      if (active >= concurrency) {
        if (waiting.length >= maxQueue) throw new QueueFullError();
        await new Promise<void>((resolve) => waiting.push(resolve));
      } else {
        active++;
      }
      try {
        return await fn();
      } finally {
        next();
      }
    },
  };
}
