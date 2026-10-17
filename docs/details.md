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
  "model": "MobileNetV2",
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

The models only look at a 224–256 px version of the image, so a ~512 px thumbnail gives the same kind of verdict for a fraction of the work: most of the time spent on a large upload is decoding it. Measured on 1 vCPU:

| Upload | 4000×3000 original | 512 px thumbnail |
| --- | --- | --- |
| WebP | 414 ms | 69 ms |
| PNG | 145 ms | 68 ms |
| JPEG | 112 ms | 63 ms |

If your app already makes resized versions of its images (an image CDN, a thumbnail pipeline), send one of those: 2 to 6× less CPU per image, less bandwidth, and lower memory peaks. Keep it at least 256 px on its shortest side.

A request that arrives while the models load waits for them (up to 60 s) instead of failing, so the first request after a cold start, or after Railway wakes a sleeping service, still gets an answer.

### `GET /` (alias `GET /health`)

`200 {"status":"ok","model":"MobileNetV2",...}` once ready, `503` while the models load. Railway's healthcheck uses `/health`.

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

Each image also goes through [MobileCLIP-S0](https://huggingface.co/Xenova/mobileclip_s0) (Apple, a CLIP model built for phones), which compares it with text descriptions listed in [`src/clip-categories.ts`](../src/clip-categories.ts): a few prompts per category (`weapon`, `gore`, `drugs`) and ~37 "other" prompts that are never flagged: everyday scenes, look-alikes (toy guns, kitchen knives, ketchup, medicine) and people/bodies, which nsfwjs judges instead. Without the people/bodies prompts, suggestive photos scored up to 39% gore and 27% drugs (measured with the previous model, CLIP ViT-B/32). A category's score is the share of CLIP's probability that goes to its prompts.

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
| `CLIP_THREADS` | `1` | ONNX Runtime threads per image. Set to the instance's vCPU count. |

To add or reword a category, edit `src/clip-categories.ts` and rebuild: prompts are turned into embeddings at build time (`npm run prepare-models`), so the 170 MB text model never ships.

Known limits: zero-shot CLIP confuses look-alikes (two white lines on black score ~74% "drugs" with MobileCLIP-S0, ~43% with ViT-B/32), misses small objects in busy scenes, and has never been evaluated on a labelled set here: 7 test images are not a benchmark.

### MobileCLIP license

The model comes from [Xenova/mobileclip_s0](https://huggingface.co/Xenova/mobileclip_s0) at a pinned revision, which ships Apple's July 2024 weights license (Apple Sample Code License: use, modification and redistribution allowed with no non-commercial restriction, as long as the notice is kept). The build copies it to `models/clip/LICENSE`, inside the image.

In August 2025 Apple moved the official MobileCLIP weights ([apple/MobileCLIP-S0](https://huggingface.co/apple/MobileCLIP-S0), [ml-mobileclip](https://github.com/apple/ml-mobileclip)) to the Apple Machine Learning Research Model license, which only allows non-commercial research. Whether the 2024 terms still cover copies obtained under them is a legal question this project does not answer. If that risk is not acceptable, go back to CLIP ViT-B/32 (MIT): ~148 MB compressed image, ~155 MB RAM idle, same verdicts on the test images.

## Cost

Docker, x86-64, 1 vCPU, the 7 test images, 2 requests at a time:

| | nsfwjs only (TensorFlow.js) | + CLIP ViT-B/32, first prototype | + CLIP ViT-B/32 q8, TensorFlow.js | + CLIP ViT-B/32 q8, ONNX only | **+ MobileCLIP-S0 (now)** |
| --- | --- | --- | --- | --- | --- |
| Compressed image | ~40 MB | ~254 MB | ~150 MB | ~148 MB | **~122 MB** |
| RAM, idle → under load | ~110 MB | ~600 MB | ~315 → 332 MB | ~155 → 170 MB | **~125 → 140 MB** |
| Per image | ~55 ms | ~147 ms | ~116 ms | ~78 ms | **~71 ms** |

The last three columns give the same verdicts on the test images. Switching to ONNX only kept every score identical to 4 decimals; MobileCLIP-S0 changes the CLIP scores (it is a different model) but is more clear-cut: 100% weapon and drugs on the two positive pictures instead of 99% and 89%, and 0.4% weapon on the drawing instead of 12%.

On Railway, an idle service mostly pays for its RAM. If the API is called rarely, turn on **App Sleeping** (service *Settings*, or `"sleepApplication": true` under `deploy` in `railway.json`): the service stops after ~10 minutes without traffic and the next request wakes it, the request waits for the container to start and the models to load (~0.75 s) instead of failing.

Why it is built this way:
- **One runtime**: nsfwjs's MobileNetV2 is converted to ONNX at build time ([`scripts/prepare-nsfw.ts`](../scripts/prepare-nsfw.ts)), keeping its uint8 weights (2.8 MB) and its exact preprocessing. TensorFlow.js and its WASM heap are gone: the nudity pass takes ~6 ms instead of ~47 ms, and its scores match nsfwjs's to ~1e-6 (checked by `test/model.test.ts`).
- **MobileCLIP-S0 fp32 (46 MB)** instead of CLIP ViT-B/32 q8 (89 MB): same speed on 1 vCPU, −27 MB compressed image, −30 MB RAM idle. `CLIP_VARIANT` (build arg) can pick `fp16` instead: 23 MB file (about −21 MB compressed), same verdicts, ~5% slower and ~10 MB more RAM, since ONNX Runtime converts it to float32 on CPU. MobileCLIP's int8 export (12 MB) is not offered: it missed both the weapon and the drugs pictures.
- **glibc malloc tuning** (`MALLOC_ARENA_MAX=2`, `MALLOC_TRIM_THRESHOLD_` and `MALLOC_MMAP_THRESHOLD_` at 128 KB): freed buffers go back to the OS. Without them, RSS settled anywhere between 254 and 301 MB from one run to the next; with them, a stable ~231 MB in the same benchmark, at no measurable latency cost.
- **Measured and rejected**: `--jitless` (the server does not start), fewer V8 or libuv threads (no difference beyond noise), turning off ONNX Runtime's memory arena (more RAM, not less), turning off weight prepacking (same RAM once malloc is tuned, +9% latency), storing the nudity model in float32 (+1 MB compressed, same RAM), decoding the image once for both models (preprocessing is ~3 ms).
- **`node --max-semi-space-size=1`**: caps V8's young generation at 1 MB per semi-space instead of 16 MB. Image buffers live outside the JS heap, so it only trims memory V8 would keep: −10 to −17 MB at peak, same latency.
- **`sharp.cache(false)`**: every upload is a new image, so libvips' operation cache (up to 50 MB) would only hold memory.
- **Debian, not Alpine**: ONNX Runtime has no musl build.
- **Distroless runtime** (`gcr.io/distroless/cc-debian12`): glibc + libstdc++ only, which is all node, sharp and ONNX Runtime link against; plus a stripped node binary (−18 MB).
- **Telemetry off**: ONNX Runtime 1.30 ships usage telemetry; `ORT_DISABLE_TELEMETRY=1` disables it.
- **Build needs network**: the MobileCLIP vision model, its license and its text model (for the prompts only) are downloaded from Hugging Face at a pinned revision during `docker build`.

What the image is made of (uncompressed): node 107 MB, ONNX Runtime 46 MB, MobileCLIP vision model 46 MB, distroless base ~25 MB, libvips 19 MB, nudity model 3 MB, server bundle 0.1 MB. Docker Desktop shows ~375 MB because it adds the compressed layers (~122 MB, what Railway downloads) to the unpacked copy.

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

Requires Node 22.18+, which runs the TypeScript sources directly (type stripping): no compile step for `dev`, the scripts or the tests. `tsc` only type-checks, and esbuild bundles the server into `dist/` for production.

```bash
npm install
npm run prepare-models  # converts the nudity model, downloads CLIP, embeds the prompts (into models/)
npm run dev             # from source, http://localhost:8080
npm run typecheck       # tsc, no emit
npm test                # type-check, then unit, real-model and production-bundle tests
npm run build           # bundle into dist/
npm start               # run dist/
```

## Notes

- The models only see a small downscale of the image (224×224 for nudity, 256×256 for MobileCLIP). It is a moderation aid, not a guarantee: keep a human in the loop for borderline cases.
- Animated GIFs are classified on their first frame.
- Nothing is written to disk; images only live in memory for the duration of the request.
