# =============================================================================
# Stage 1: Builder
# =============================================================================
# Separate build and runtime stages to prevent build-time dependencies
# (cmake, gcc, headers) from inflating the final image size. The compiled
# llama-cpp-python wheel is the only artifact transferred to the runtime stage.
#
# Base image reproducibility: both stages pin the Debian release in the tag
# (slim-trixie) and the exact multi-arch index by digest. The plain 3.11-slim
# tag moves on every upstream rebuild and silently switched Debian releases
# once already; with a digest, a base change only happens through a reviewed
# commit (Dependabot proposes digest bumps, see .github/dependabot.yml).
# Keep the two FROM lines identical so builder and runtime share a libc.
FROM python:3.11-slim-trixie@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce AS builder

# Build dependencies for llama-cpp-python's C++ compilation.
# libopenblas-dev provides the BLAS backend used for matrix operations —
# the primary compute path for transformer attention on CPU.
#
# apt packages are deliberately not version-pinned (DL3008). Debian removes
# superseded versions from its mirrors when security updates land, so exact
# pins break the build precisely when a patched package is available. The
# digest-pinned base fixes the release; packages track its security updates.
# hadolint ignore=DL3008
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
# CPU portability: llama.cpp defaults to GGML_NATIVE=ON, i.e. -march=native,
# which compiles for whatever CPU happens to run the build. An image built on
# a CI runner with AVX-512 would then die with SIGILL on a node without it,
# and only on that node. GGML_NATIVE=OFF with an explicit instruction set
# pins the x86-64-v3 baseline (AVX2/FMA/F16C, Intel Haswell / AMD Zen and
# newer), which every current cloud and desktop x86 CPU supports. AVX-512 is
# left off on purpose: its speedup for quantized matmul is modest and it is
# absent on many consumer and some cloud CPUs.
#
# GPU transition path: replace with CMAKE_ARGS="-DGGML_CUDA=on", switch to a
# CUDA base image, and add --gpus flags to offload layers to VRAM.
ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS \
-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
-DGGML_AVX512=OFF"
ENV FORCE_CMAKE=1

# pip is not upgraded first: an unpinned `pip install --upgrade pip` made
# the build depend on whatever pip release was current that day. The pip
# shipped in the digest-pinned base image builds these wheels fine.
RUN pip wheel --no-cache-dir --wheel-dir=/dist -r requirements.txt
# Single pip wheel invocation covers all dependencies including llama-cpp-python.
# CMAKE_ARGS and FORCE_CMAKE set above are inherited here, ensuring the C++
# compilation uses OpenBLAS. Running llama-cpp-python as a separate step
# first caused version conflicts when GHA layer cache held a stale wheel.


# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim-trixie@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce AS runtime

# PYTHONDONTWRITEBYTECODE: the root filesystem is read-only in the pod, so
# bytecode writes would fail on every import anyway.
# PYTHONUNBUFFERED: log lines reach the container runtime as they are
# written, so nothing is lost in a buffer when a pod is OOM-killed.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Only the shared libraries are needed at runtime — not the dev headers or
# the compiler toolchain. libgomp1 is the OpenMP runtime: llama.cpp builds
# with OpenMP threading by default, and the builder gets it implicitly from
# build-essential, so its absence here only shows up when libllama.so loads.
# Unpinned for the same reason as in the builder stage.
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pre-built wheels from the builder stage.
# This avoids re-compiling llama-cpp-python in CI and keeps layer caching
# efficient — the wheels layer only invalidates when requirements.txt changes.
#
# setuptools and wheel ship with the python base image but nothing imports
# them at runtime. They are removed rather than upgraded: setuptools vendors
# its own copies of wheel and jaraco.context, which image scanners report as
# fixable HIGH CVEs, and code that is not present cannot be exploited.
#
# llama-cpp-python 0.3 wheels also install a second copy of every native
# library, plus CMake and pkg-config development files, into a top-level
# site-packages/lib directory. The package only loads from llama_cpp/lib
# (none of those libraries link against the extra copy), so the duplicate
# is dropped to keep ~37MB of unused shared objects out of the image.
COPY --from=builder /dist /dist
RUN pip install --no-cache-dir /dist/*.whl \
    && rm -rf /dist \
    && pip uninstall --yes setuptools wheel \
    && site_packages="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && rm -rf "${site_packages:?}/lib"

# Build-time guard: load the native libraries and fail the build if the BLAS
# backend is not registered, or if a shared library the wheel depends on is
# missing from this slimmer runtime image. Both would otherwise surface only
# when the pod starts in the cluster.
#
# The check queries ggml's backend registry directly instead of parsing
# llama_print_system_info(). Since llama-cpp-python 0.3, ggml backends are
# separate libraries, and the system info string only lists backends that
# export a feature table; the BLAS backend does not, so it is absent from
# that output even when compiled in and active.
RUN python -c "import ctypes, os, sys, llama_cpp; \
ggml = ctypes.CDLL(os.path.join(os.path.dirname(llama_cpp.__file__), 'lib', 'libggml.so')); \
ggml.ggml_backend_reg_count.restype = ctypes.c_size_t; \
ggml.ggml_backend_reg_get.argtypes = [ctypes.c_size_t]; \
ggml.ggml_backend_reg_get.restype = ctypes.c_void_p; \
ggml.ggml_backend_reg_name.argtypes = [ctypes.c_void_p]; \
ggml.ggml_backend_reg_name.restype = ctypes.c_char_p; \
names = [ggml.ggml_backend_reg_name(ggml.ggml_backend_reg_get(i)).decode() for i in range(ggml.ggml_backend_reg_count())]; \
print('ggml backends:', names); \
print(llama_cpp.llama_print_system_info().decode()); \
sys.exit(0 if 'BLAS' in names else 'llama.cpp was built without the BLAS backend')"

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

# Numeric UID:GID rather than the name: the kubelet can only verify
# runAsNonRoot against a numeric user, and it matches runAsUser in the
# Deployment without relying on /etc/passwd inside the image.
USER 1000:1000

EXPOSE 8080

# workers=1 is non-negotiable: the Llama model instance is process-local.
# Multiple uvicorn workers would each load a separate copy of the model,
# exceeding the 4Gi memory limit and triggering an OOM kill from the kubelet.
# Horizontal scaling is handled externally by KEDA.
# --no-access-log: the application writes its own access log with request IDs
# (app/logging_config.py); uvicorn's would duplicate every line without them.
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]
