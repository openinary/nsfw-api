export class QueueFullError extends Error {
  constructor() {
    super("Server busy, retry later");
  }
}

// Caps how many inferences run at once and how many may wait behind them.
export function createLimiter({ concurrency, maxQueue }) {
  let active = 0;
  const waiting = [];

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
        await new Promise((resolve) => waiting.push(resolve));
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
