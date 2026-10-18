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
# The GGML_ prefix is required: llama.cpp renamed its build options from
# LLAMA_* to GGML_* and llama-cpp-python 0.2.80+ silently ignores the old
# names, producing a working but BLAS-less binary. The runtime stage asserts
# that BLAS was actually linked so this cannot regress unnoticed.
#
# GPU transition path: replace with CMAKE_ARGS="-DGGML_CUDA=on", switch to a
# CUDA base image, and add --gpus flags to offload layers to VRAM.
ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
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

# Only the shared libraries are needed at runtime — not the dev headers or
# the compiler toolchain. libgomp1 is the OpenMP runtime: llama.cpp builds
# with OpenMP threading by default, and the builder gets it implicitly from
# build-essential, so its absence here only shows up when libllama.so loads.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pre-built wheels from the builder stage.
# This avoids re-compiling llama-cpp-python in CI and keeps layer caching
# efficient — the wheels layer only invalidates when requirements.txt changes.
COPY --from=builder /dist /dist
RUN pip install --no-cache-dir /dist/*.whl && rm -rf /dist

# Build-time guard: load the native library and fail the build if the BLAS
# backend is not compiled in, or if a shared library the wheel depends on is
# missing from this slimmer runtime image. Both would otherwise surface only
# when the pod starts in the cluster.
RUN python -c "import sys, llama_cpp; \
info = llama_cpp.llama_print_system_info().decode(); \
print(info); \
sys.exit(0 if 'BLAS = 1' in info else 'llama.cpp was built without BLAS support')"

COPY app/ ./app/

# Non-root user with a fixed UID for predictable SecurityContext behavior.
# Kubernetes PodSecurityPolicy / Pod Security Standards (restricted) require
# runAsNonRoot=true; this satisfies that constraint without image rebuilds.
RUN useradd --create-home --uid 1000 --shell /bin/sh vortex

# Model files are mounted via PersistentVolumeClaim at /models.
# The directory is created here so the mount target exists before the
# volume is attached — Kubernetes will not create the path automatically
# on hostPath or NFS mounts if it does not exist in the image.
# Must run before USER: an unprivileged user cannot create directories in /.
RUN mkdir -p /models && chown vortex:vortex /models

USER vortex

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
