# Details

Everything the [README](../README.md) leaves out: the full API, every setting, how the CLIP categories work, what it costs to run, and why it is built this way.

## API

### `POST /classify`

Send the image as `multipart/form-data` (field `image`) or as a raw body.

```bash
curl -X POST https://your-app.up.railway.app/classify \
  -H "Authorization: Bearer $API_KEY" \
  -F image=@photo.jpg
```

```bash
curl -X POST "https://your-app.up.railway.app/classify?threshold=0.5" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: image/webp" \
  --data-binary @photo.webp
```

Response:

```json
{
  "safe": true,
  "flags": [],
  "nsfw": false,
  "score": 0.0036,
  "threshold": 0.7,
  "top": { "className": "Neutral", "probability": 0.713 },
  "predictions": { "Drawing": 0.284, "Hentai": 0.003, "Neutral": 0.713, "Porn": 0, "Sexy": 0 },
  "categories": {
    "weapon": { "score": 0.002, "match": null, "flagged": false },
    "gore": { "score": 0.001, "match": null, "flagged": false },
    "drugs": { "score": 0.001, "match": null, "flagged": false }
  },
  "model": "MobileNetV2Mid",
  "timeMs": 76
}
```

`safe` is `true` when `flags` is empty. `score` is the sum of the classes listed in `NSFW_CLASSES`, and `nsfw` is `score > threshold`. `?threshold=` overrides `NSFW_THRESHOLD` per request.

| Status | Meaning |
| --- | --- |
| 400 | No image in the request, or invalid `threshold` |
| 401 | `API_KEY` is set and the request has no valid key |
| 413 | Image larger than `MAX_FILE_SIZE_MB` |
| 415 | Unsupported or corrupt image |
| 429 | Too many requests queued (`Retry-After: 1`) |
| 503 | Models still loading after 60 s |

#### Send a thumbnail, not the original

The models only look at a 224–256 px version of the image, so a ~512 px thumbnail gives the same kind of verdict for a fraction of the work: most of the time spent on a large upload is decoding it. Measured on 1 vCPU (arm64):

| Upload | 4000×3000 original | 512 px thumbnail |
| --- | --- | --- |
| WebP | 207 ms | 94 ms |
| PNG | 185 ms | 92 ms |
| JPEG | 130 ms | 92 ms |

If your app already makes resized versions of its images (an image CDN, a thumbnail pipeline), send one of those: up to 2× less CPU per image, less bandwidth, and lower memory peaks (a large upload is decoded whole in memory, 3 to 4 bytes per pixel). Keep it at least 256 px on its shortest side.

A request that arrives while the models load waits for them (up to 60 s) instead of failing, so the first request after a cold start, or after Railway wakes a sleeping service, still gets an answer.

### `GET /` (alias `GET /health`)

`200 {"status":"ok","model":"MobileNetV2Mid",...}` once ready, `503` while the models load. Railway's healthcheck uses `/health`.

### `GET /playground`

Browser page to try the API by hand. Turn it off with `ENABLE_PLAYGROUND=false`.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `API_KEY` | _(empty)_ | Required as `Authorization: Bearer <key>` or `x-api-key`. Leave empty for an open API. |
| `NSFW_THRESHOLD` | `0.7` | Score above which `nsfw` is `true`. |
| `NSFW_CLASSES` | `Porn,Hentai,Sexy` | Classes summed into `score`. Use `Porn,Hentai` to tolerate suggestive images. |
| `MAX_FILE_SIZE_MB` | `10` | Upload size limit. |
| `MAX_CONCURRENCY` | `2` | Images analysed at once. |
| `MAX_QUEUE` | `32` | Requests allowed to wait; beyond that the API returns 429. |
| `ENABLE_PLAYGROUND` | `true` | Serve the playground at `/playground`. |
| `CORS_ORIGIN` | _(empty)_ | Comma-separated origins allowed to call the API from a browser. |
| `PORT` | `8080` | Set automatically by Railway. |

## CLIP categories (prototype)

Each image also goes through [MobileCLIP-S0](https://huggingface.co/Xenova/mobileclip_s0) (Apple, a CLIP model built for phones), which compares it with text descriptions listed in [`scripts/clip-categories.ts`](../scripts/clip-categories.ts): a few prompts per category (`weapon`, `gore`, `drugs`) and ~37 "other" prompts that are never flagged: everyday scenes, look-alikes (toy guns, kitchen knives, ketchup, medicine) and people/bodies, which nsfwjs judges instead. Without the people/bodies prompts, suggestive photos scored up to 39% gore and 27% drugs (measured with the previous model, CLIP ViT-B/32). A category's score is the share of CLIP's probability that goes to its prompts.

```json
"safe": false,
"flags": ["weapon"],
"categories": {
  "weapon": { "score": 0.91, "flagged": true, "match": "a person holding a gun" },
  "gore":   { "score": 0.02, "flagged": false, "match": "a photo of a bloody wound" }
}
```

| Variable | Default | Description |
| --- | --- | --- |
| `CLIP_CATEGORIES` | `weapon,gore,drugs` | Categories to check. `none` turns CLIP off (it still ships in the image). |
| `CLIP_THRESHOLD` | `0.5` | Score above which a category is flagged. Not calibrated yet. |

To add or reword a category, edit `scripts/clip-categories.ts` and rebuild: prompts are turned into embeddings at build time (`npm run prepare-models`), so the 170 MB text model never ships.

Known limits: zero-shot CLIP confuses look-alikes (two white lines on black score ~74% "drugs" with MobileCLIP-S0, ~43% with ViT-B/32), misses small objects in busy scenes, and has never been evaluated on a labelled set here: 7 test images are not a benchmark.

### MobileCLIP license

The model comes from [Xenova/mobileclip_s0](https://huggingface.co/Xenova/mobileclip_s0) at a pinned revision, which ships Apple's July 2024 weights license (Apple Sample Code License: use, modification and redistribution allowed with no non-commercial restriction, as long as the notice is kept). The build copies it to `models/clip/LICENSE`, inside the image.

In August 2025 Apple moved the official MobileCLIP weights ([apple/MobileCLIP-S0](https://huggingface.co/apple/MobileCLIP-S0), [ml-mobileclip](https://github.com/apple/ml-mobileclip)) to the Apple Machine Learning Research Model license, which only allows non-commercial research. Whether the 2024 terms still cover copies obtained under them is a legal question this project does not answer. If that risk is not acceptable, go back to CLIP ViT-B/32 (MIT): ~148 MB compressed image, ~155 MB RAM idle, same verdicts on the test images.

## Cost

Version 2 is a Rust server running the models with [tract](https://github.com/sonos/tract) instead of Node with ONNX Runtime. Both versions measured the same way: Docker, arm64 (Apple Silicon), 1 vCPU, a mix of 7 JPEG, PNG and WebP photos from 512 to 1200 px, 300 requests:

| | v1: Node + ONNX Runtime | **v2: Rust + tract (now)** |
| --- | --- | --- |
| Image, compressed (unpacked) | 113 MB (236 MB) | **66 MB (115 MB)** |
| RAM, idle after start | 128 MiB | **67 MiB** |
| RAM under load | 175 MiB | **146 MiB** |
| RAM back idle, 15 s after the load | 172 MiB | **103 MiB** |
| Per image, 1 request at a time | 120 ms | **93 ms** |
| Per image, 2 requests at a time (throughput) | 120 ms | **114 ms** |
| Container start to healthy | 0.59 s | 0.65 s |

Scores moved by at most 0.005 between the two versions on 13 test images (the resize is no longer libvips'); verdicts are unchanged. Not measured yet: x86-64 on real hardware. Emulated on the same machine, v2 uses less memory there too (106 → 190 MiB against 257 → 313) and is as fast one request at a time, but its throughput at 2 requests at a time is ~25% lower (390 ms per image against 310). Emulation distorts timings, so measure on your own x86-64 instance before relying on either figure.

On Railway, an idle service mostly pays for its RAM. If the API is called rarely, turn on **App Sleeping** (service *Settings*, or `"sleepApplication": true` under `deploy` in `railway.json`): the service stops after ~10 minutes without traffic and the next request wakes it, the request waits for the container to start and the models to load (~0.7 s) instead of failing.

Why it is built this way:
- **One binary**: the HTTP server (axum), image decoding (the `image` crate, resvg for SVG), resizing (`fast_image_resize`) and inference (tract) are compiled into a single ~25 MB executable. No Node, no ONNX Runtime, no libvips: that is where the image size and the idle memory went.
- **Models prepared at build time, by Node**: nsfwjs's MobileNetV2Mid is converted to ONNX ([`scripts/prepare-nsfw.ts`](../scripts/prepare-nsfw.ts)), keeping its exact preprocessing, and the CLIP prompts are embedded with transformers.js ([`scripts/prepare-clip.ts`](../scripts/prepare-clip.ts)). Both are npm packages, so that stage stays in Node; nothing of it ships. The nudity scores match nsfwjs's own to 4 decimals (checked by `test/e2e.test.ts`).
- **MobileCLIP-S0 fp32 (46 MB)** instead of CLIP ViT-B/32 q8 (89 MB). `CLIP_VARIANT` (build arg) can pick `fp16` instead: a 23 MB file, which loads and classifies but has not been benchmarked on tract. MobileCLIP's int8 export (12 MB) is not offered: it missed both the weapon and the drugs pictures.
- **The image is decoded once** for both models, which then run side by side on two threads.
- **Memory is handed back when idle**: glibc keeps freed memory in its heap, ~100 MB of scratch space after optimising the models and ~75 MB of tensors after a burst. The server calls `malloc_trim` once the models are loaded and again after 5 quiet seconds. `MALLOC_ARENA_MAX=2` keeps freed tensors reusable across threads (−10 MB under load, same speed).
- **Measured and rejected**: `MALLOC_TRIM_THRESHOLD_=131072`, which trims on every free. It holds RAM at ~70 → 86 MiB even under load, but costs ~30% of the throughput (157 ms per image instead of 114). Set it on the service if you would rather pay CPU than RAM.
- **Debian 13, not Alpine**: AVIF is decoded by libdav1d, the one shared library the binary needs, and the Rust bindings want a more recent version than Debian 12 has.
- **Distroless runtime** (`gcr.io/distroless/cc-debian13`): glibc only, no shell, no package manager, runs as `nonroot`.
- **Build needs network**: the MobileCLIP vision model, its license and its text model (for the prompts only) are downloaded from Hugging Face at a pinned revision during `docker build`.

What the image is made of (uncompressed): MobileCLIP vision model 46 MB, distroless base ~38 MB, server binary 25 MB, nudity model 4.5 MB, libdav1d 0.7 MB.

<details>
<summary>History: the Node versions (x86-64, 1 vCPU, 7 test images, 2 requests at a time)</summary>

| | nsfwjs only (TensorFlow.js) | + CLIP ViT-B/32, first prototype | + CLIP ViT-B/32 q8, TensorFlow.js | + CLIP ViT-B/32 q8, ONNX only | + MobileCLIP-S0, ONNX Runtime (v1) |
| --- | --- | --- | --- | --- | --- |
| Compressed image | ~40 MB | ~254 MB | ~150 MB | ~148 MB | ~122 MB |
| RAM, idle → under load | ~110 MB | ~600 MB | ~315 → 332 MB | ~155 → 170 MB | ~125 → 140 MB |
| Per image | ~55 ms | ~147 ms | ~116 ms | ~78 ms | ~71 ms |

These were measured on another machine and another set of images, so they do not compare with the table above.

</details>

## Testing by hand before deploying

Run the exact image Railway will build, with the same limits as a small instance:

```bash
docker build -t nsfw-api .
docker run --rm -p 8080:8080 --cpus=1 -m 1g -e API_KEY=test-key nsfw-api
```

1. Open <http://localhost:8080/playground>. The status line should say **Ready**.
2. Paste `test-key` in the API key field.
3. Click *Use a generated test picture*: expect **Safe**.
4. Drop your own images, including ones you expect to be flagged, and check the verdicts. Move the threshold slider to see where you'd want `NSFW_THRESHOLD`.
5. Try the edge cases: a wrong key (error 401), a non-image file such as a PDF (error 415), an image over 10 MB (error 413).
6. From another terminal, run the automated end-to-end checks against it:

   ```bash
   API_KEY=test-key npm run smoke -- http://localhost:8080
   ```

The same smoke test works against the deployed URL once it is on Railway.

## Development

The server is Rust (`src/`, stable toolchain). Node 22.18+ only prepares the models (`scripts/`, where nsfwjs and transformers.js live) and runs the end-to-end tests; it never ships in the image.

```bash
npm install
npm run prepare-models  # converts the nudity model, downloads CLIP, embeds the prompts (into models/)
npm run dev             # cargo run --release, http://localhost:8080
cargo test              # unit tests and the HTTP layer against a stub classifier
npm test                # cargo test, then end-to-end tests on the release binary with the real models
```

AVIF decoding is a cargo feature because it links the system's libdav1d: the Dockerfile turns it on, a plain `cargo build` leaves it out (AVIF uploads then get a 415). To build and test it locally, install `pkg-config` and libdav1d, then run `cargo build --release --features avif` and `TEST_AVIF=1 npx vitest run`.

## Notes

- The models only see a small downscale of the image (224×224 for nudity, 256×256 for MobileCLIP). It is a moderation aid, not a guarantee: keep a human in the loop for borderline cases.
- Animated GIFs are classified on their first frame.
- TIFF: uncompressed, LZW and Deflate RGB or greyscale files are read; JPEG-compressed YCbCr TIFFs get a 415.
- SVGs are rendered without text and without the images they embed or link to.
- Nothing is written to disk; images only live in memory for the duration of the request.
