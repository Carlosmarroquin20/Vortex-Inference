"""
Configuration management via Pydantic Settings.

Design rationale: All tunables are environment-variable-driven to align with
the 12-factor app methodology. This allows the same container image to be
promoted across environments (dev → staging → prod) by changing only the
ConfigMap or Secret — not the image layer. No hardcoded paths or magic numbers.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Runtime configuration for the inference server.

    Values are resolved in this priority order:
      1. Environment variables (injected by Kubernetes ConfigMap/Secret)
      2. .env file (local development only, never committed)
      3. Defaults defined here (safe fallbacks for resource-constrained environments)
    """

    # --- Model Configuration ---
    model_path: str = "/models/phi-3-mini-4k-instruct-q4_k_m.gguf"
    model_name: str = "phi-3-mini-4k-instruct"

    # n_ctx defines the KV-cache allocation at startup. Setting this above the
    # model's actual training context (4096 for Phi-3-mini) wastes VRAM/RAM
    # without any quality benefit. In memory-constrained pods, this is the
    # first lever to pull if the process approaches its memory limit.
    n_ctx: int = 4096

    # n_threads should match the pod's CPU request, not its limit.
    # Using the limit value risks throttling during burst inference, which
    # causes latency spikes that look like application bugs in dashboards.
    # The relationship: n_threads=2 → ~1.5–2 tok/s on modern x86 with OpenBLAS.
    n_threads: int = 2

    # n_gpu_layers=0 enforces full CPU inference. In a GPU-enabled cluster,
    # set to -1 to offload all layers to VRAM and gain ~10x throughput.
    # This single flag is the only change needed to transition from CPU to GPU.
    n_gpu_layers: int = 0

    # max_concurrent_requests=1 is deliberate for single-pod CPU inference.
    # llama.cpp processes tokens sequentially in a single C++ thread pool.
    # Parallel requests don't increase throughput — they increase memory
    # pressure and produce degraded output for all in-flight requests.
    # KEDA handles concurrency by scaling horizontally, not vertically.
    max_concurrent_requests: int = 1

    # --- Server Configuration ---
    port: int = 8080

    # Inference timeout in seconds. Must be shorter than the Kubernetes
    # Service's session affinity timeout and the client's read timeout.
    # Tuned for worst-case Phi-3-mini on 2 CPU threads at 2048 max_tokens.
    inference_timeout_seconds: float = 120.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VORTEX_",
        case_sensitive=False
    )
