// Bundles the server (Hono included) into dist/, so the image only needs the
// native addons from node_modules.
import { copyFile, rm } from "node:fs/promises";
import { build } from "esbuild";

await rm("dist", { recursive: true, force: true });

await build({
  entryPoints: ["src/server.js"],
  outdir: "dist",
  bundle: true,
  format: "esm",
  platform: "node",
  target: "node22",
  minify: true,
  legalComments: "none",
  // Native addons stay as installed packages.
  external: ["sharp", "onnxruntime-node"],
  logLevel: "warning",
});

await copyFile("src/playground.html", "dist/playground.html");

console.log("Built dist/");
