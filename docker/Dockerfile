# =============================================================================
# Stage 1: Builder
# =============================================================================
# Separate build and runtime stages to prevent build-time dependencies
# (cmake, gcc, headers) from inflating the final image size. The compiled
# llama-cpp-python wheel is the only artifact transferred to the runtime stage.
FROM python:3.11-slim AS builder

# Build dependencies for llama-cpp-python's C++ compilation.
# libopenblas-dev provides the BLAS backend used for matrix operations —
# the primary compute path for transformer attention on CPU.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# CMAKE_ARGS enables the OpenBLAS BLAS backend during llama.cpp compilation.
# Without this flag, the build falls back to naive C matrix multiplication,
# which is approximately 1.5–2x slower for the matrix-vector products
# dominating autoregressive token generation on x86 CPUs.
#
# GPU transition path: replace with CMAKE_ARGS="-DLLAMA_CUBLAS=on" and add
# --gpus flags to the container runtime to offload layers to VRAM.
ENV CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
ENV FORCE_CMAKE=1

RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir=/dist -r requirements.txt
# Single pip wheel invocation covers all dependencies including llama-cpp-python.
# CMAKE_ARGS and FORCE_CMAKE set above are inherited here, ensuring the C++
# compilation uses OpenBLAS. Running llama-cpp-python as a separate step
# first caused version conflicts when GHA layer cache held a stale wheel.


# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim AS runtime

# Only the shared library for OpenBLAS is needed at runtime — not the
# dev headers or the compiler toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pre-built wheels from the builder stage.
# This avoids re-compiling llama-cpp-python in CI and keeps layer caching
# efficient — the wheels layer only invalidates when requirements.txt changes.
COPY --from=builder /dist /dist
RUN pip install --no-cache-dir /dist/*.whl && rm -rf /dist

COPY app/ ./app/

# Non-root user with a fixed UID for predictable SecurityContext behavior.
# Kubernetes PodSecurityPolicy / Pod Security Standards (restricted) require
# runAsNonRoot=true; this satisfies that constraint without image rebuilds.
RUN useradd --create-home --uid 1000 --shell /bin/sh vortex
USER vortex

# Model files are mounted via PersistentVolumeClaim at /models.
# The directory is created here so the mount target exists before the
# volume is attached — Kubernetes will not create the path automatically
# on hostPath or NFS mounts if it does not exist in the image.
RUN mkdir -p /models

EXPOSE 8080

# workers=1 is non-negotiable: the Llama model instance is process-local.
# Multiple uvicorn workers would each load a separate copy of the model,
# exceeding the 4Gi memory limit and triggering an OOM kill from the kubelet.
# Horizontal scaling is handled externally by KEDA.
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--log-level", "info"]
