# Debian (glibc) because ONNX Runtime's native binaries don't exist for Alpine/musl.
FROM node:22-bookworm-slim AS build
WORKDIR /app
# Skip ONNX Runtime's optional CUDA download.
ENV ONNXRUNTIME_NODE_INSTALL=skip
RUN apt-get update && apt-get install -y --no-install-recommends binutils && rm -rf /var/lib/apt/lists/* \
 # The official binary ships debug symbols: stripping saves ~18 MB.
 && strip /usr/local/bin/node
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY build.ts ./
COPY src ./src
COPY scripts/prepare-nsfw.ts scripts/prepare-clip.ts ./scripts/
# Nudity: convert nsfwjs's MobileNetV2 (installed as a dev dependency) to ONNX.
RUN node scripts/prepare-nsfw.ts
# Railway passes service variables as build args.
# CLIP (MobileCLIP-S0) weights precision: fp32 (default, 46 MB) or fp16 (23 MB, ~5% slower).
ARG CLIP_VARIANT=fp32
# CLIP: download the vision model and embed the category prompts (needs network at build time).
RUN node scripts/prepare-clip.ts && rm -rf models/.cache
RUN node build.ts \
 && npm prune --omit=dev \
 # Keep only this platform's native binaries.
 && find node_modules/@img -mindepth 1 -maxdepth 1 \( -name "*-linuxmusl-*" -o -name "*-wasm32" \) -exec rm -rf {} + \
 && find node_modules/onnxruntime-node/bin/napi-v6 -mindepth 1 -maxdepth 1 ! -name linux -exec rm -rf {} + \
 && find node_modules/onnxruntime-node/bin/napi-v6/linux -mindepth 1 -maxdepth 1 ! -name "$(node -p process.arch)" -exec rm -rf {} + \
 && find node_modules -name "*.map" -type f -delete \
 && find node_modules -mindepth 1 -maxdepth 1 -type d -empty -delete

# Runtime: distroless "cc" is just glibc + libstdc++ (~25 MB), all that node,
# sharp and ONNX Runtime link against. No shell, no package manager; runs as nonroot.
FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=build /usr/local/bin/node /usr/local/bin/node
WORKDIR /app
# ONNX Runtime ships usage telemetry; keep a moderation API from phoning home.
# glibc malloc tuning, measured on this image (1 vCPU, 2 concurrent requests):
#  - MALLOC_ARENA_MAX=2: per-thread arenas cost ~50 MB under load with ORT's threads;
#  - TRIM/MMAP thresholds at 128 KB: big buffers (decoded images, model loading
#    scratch space) go back to the OS once freed. Steady RSS drops from 254-301 MB
#    (it varied run to run) to a stable ~231 MB, at no measurable latency cost.
ENV NODE_ENV=production ORT_DISABLE_TELEMETRY=1 MALLOC_ARENA_MAX=2 \
    MALLOC_TRIM_THRESHOLD_=131072 MALLOC_MMAP_THRESHOLD_=131072
COPY --from=build /app/package.json ./
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY --from=build /app/models ./models
EXPOSE 8080
# --max-semi-space-size=1: caps V8's young generation at 1 MB per semi-space
# (default 16 MB). Image buffers live outside the JS heap, so this only trims
# memory V8 would hold on to: −10 to −17 MB at peak, same latency (measured).
ENTRYPOINT ["/usr/local/bin/node", "--max-semi-space-size=1"]
CMD ["dist/server.js"]
