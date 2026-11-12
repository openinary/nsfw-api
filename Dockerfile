# Models: built with Node (nsfwjs and transformers.js are npm packages), never shipped.
FROM node:22.23.4-bookworm-slim AS models
WORKDIR /app
# Skip ONNX Runtime's optional CUDA download.
ENV ONNXRUNTIME_NODE_INSTALL=skip
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY scripts/prepare-nsfw.ts scripts/prepare-clip.ts scripts/clip-categories.ts ./scripts/
# Nudity: convert nsfwjs's MobileNetV2Mid (a dev dependency) to ONNX.
RUN node scripts/prepare-nsfw.ts
# Railway passes service variables as build args.
# CLIP (MobileCLIP-S0) weights precision: fp32 (default, 46 MB) or fp16 (23 MB).
ARG CLIP_VARIANT=fp32
# CLIP: download the vision model and embed the category prompts (needs network at build time).
RUN node scripts/prepare-clip.ts && rm -rf models/.cache

# Server: one Rust binary. Debian 13 for a libdav1d (AVIF) recent enough for the bindings.
FROM rust:1-slim-trixie AS build
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libdav1d-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked --features avif \
 # The only shared library the binary needs beyond glibc.
 && cp "$(ldd target/release/nsfw-api | awk '/libdav1d/ { print $3 }')" /libdav1d.so

# Runtime: distroless "cc" is just glibc + libgcc. No shell, no package manager; runs as nonroot.
FROM gcr.io/distroless/cc-debian13:nonroot
WORKDIR /app
COPY --from=build /libdav1d.so /usr/lib/libdav1d.so.7
COPY --from=build /app/target/release/nsfw-api ./
COPY --from=models /app/models ./models
# glibc gives each thread its own malloc arena; capping them keeps freed tensors
# reusable across the inference threads (−10 MB under load, same speed, measured).
ENV MALLOC_ARENA_MAX=2
EXPOSE 8080
# Already the base image's default; stated so scanners can see it.
USER nonroot
ENTRYPOINT ["/app/nsfw-api"]
