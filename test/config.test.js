import { describe, expect, it } from "vitest";
import { loadConfig } from "../src/config.js";

describe("loadConfig", () => {
  it("has sensible defaults", () => {
    expect(loadConfig({})).toMatchObject({
      port: 8080,
      apiKey: "",
      threshold: 0.7,
      nsfwClasses: ["Porn", "Hentai", "Sexy"],
      maxFileSize: 10 * 1024 * 1024,
      enablePlayground: true,
    });
  });

  it("reads Railway-style env vars", () => {
    const config = loadConfig({
      PORT: "3000",
      API_KEY: "k",
      NSFW_THRESHOLD: "0.5",
      NSFW_CLASSES: "Porn, Hentai",
      MAX_FILE_SIZE_MB: "2",
      ENABLE_PLAYGROUND: "false",
    });
    expect(config).toMatchObject({
      port: 3000,
      apiKey: "k",
      threshold: 0.5,
      nsfwClasses: ["Porn", "Hentai"],
      maxFileSize: 2 * 1024 * 1024,
      enablePlayground: false,
    });
  });

  it("enables CLIP categories by default and can disable them", () => {
    expect(loadConfig({})).toMatchObject({ clipCategories: ["weapon", "gore", "drugs"], clipThreshold: 0.5, clipThreads: 1 });
    expect(loadConfig({ CLIP_CATEGORIES: "weapon, gore" }).clipCategories).toEqual(["weapon", "gore"]);
    expect(loadConfig({ CLIP_CATEGORIES: "none" }).clipCategories).toEqual([]);
    expect(loadConfig({ CLIP_CATEGORIES: "" }).clipCategories).toEqual([]);
  });

  it.each([
    [{ CLIP_THRESHOLD: "2" }, /CLIP_THRESHOLD/],
    [{ NSFW_THRESHOLD: "1.5" }, /NSFW_THRESHOLD/],
    [{ NSFW_CLASSES: "Porn,Gore" }, /Gore/],
    [{ PORT: "abc" }, /PORT/],
  ])("fails fast on invalid %o", (env, message) => {
    expect(() => loadConfig(env)).toThrow(message);
  });
});
