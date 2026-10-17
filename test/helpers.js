import sharp from "sharp";
import { loadConfig } from "../src/config.js";

// Synthetic images generated on the fly, so the repo ships no binary fixtures.
export async function makeImage(format = "jpeg", { width = 640, height = 480, alpha = false } = {}) {
  const channels = alpha ? 4 : 3;
  const data = Buffer.alloc(width * height * channels);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * channels;
      data[i] = (x / width) * 255;
      data[i + 1] = (y / height) * 255;
      data[i + 2] = 160;
      if (alpha) data[i + 3] = x < width / 2 ? 255 : 0;
    }
  }
  return sharp(data, { raw: { width, height, channels } }).toFormat(format).toBuffer();
}

export function testConfig(overrides = {}) {
  return { ...loadConfig({}), ...overrides };
}

export function multipart(buffer, { field = "image", type = "image/jpeg", name = "photo.jpg" } = {}) {
  const form = new FormData();
  form.append(field, new Blob([buffer], { type }), name);
  return form;
}
