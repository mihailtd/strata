# ==============================================================================
# Strata Containerfile (AMD ROCm / HIP)
# ==============================================================================
# Multi-stage build producing a slim, production-grade serving image.
#
# Build example:
#   docker build -f apps/runtime-next/Dockerfile -t strata:latest .
#
# Run example:
#   docker run --device=/dev/kfd --device=/dev/dri --group-add video \
#     -p 8003:8003 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     strata:latest --port 8003
# ==============================================================================

# Stage 1: Build Environment
FROM rocm/dev-ubuntu-24.04:latest AS builder

ARG RUNTIME_NEXT_OFFLOAD_ARCH="gfx1100"
ARG STRATA_FEATURE="qwen35_4b"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev hipblaslt-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY apps/runtime-next/Cargo.toml apps/runtime-next/Cargo.lock apps/runtime-next/build.rs ./
COPY apps/runtime-next/src/ ./src/

# Compile Strata with the selected model feature and offload architecture
RUN RUNTIME_NEXT_OFFLOAD_ARCH="${RUNTIME_NEXT_OFFLOAD_ARCH}" \
    cargo build --release --no-default-features --features "${STRATA_FEATURE}"

# Stage 2: Runtime Image
FROM rocm/rocm-terminal:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    hipblaslt ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy executable and compiled HIP kernel shared library
COPY --from=builder /build/target/release/strata /app/strata
COPY --from=builder /build/target/release/build/strata-*/out/libruntime_next_kernels.so /app/libruntime_next_kernels.so

ENV LD_LIBRARY_PATH="/app:/opt/rocm/lib:${LD_LIBRARY_PATH}"
EXPOSE 8003

ENTRYPOINT ["/app/strata"]
CMD ["--port", "8003"]

