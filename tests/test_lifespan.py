import asyncio

import pytest

import app.main as main


def test_startup_fails_fast_when_the_backend_cannot_start(monkeypatch):
    # The process must exit rather than serve without a backend, so Kubernetes
    # applies CrashLoopBackOff instead of routing traffic to a broken pod.
    def failing_factory(settings):
        raise MemoryError("cannot allocate KV cache")

    monkeypatch.setattr(main, "create_backend", failing_factory)

    async def start():
        async with main.lifespan(main.app):
            pass

    with pytest.raises(
        RuntimeError, match="cannot start without a working llama_cpp backend"
    ):
        asyncio.run(start())


def test_model_is_loaded_with_configured_parameters(fake_model):
    assert fake_model.init_kwargs == {
        "model_path": "/models/test-model.gguf",
        "n_ctx": 64,
        "n_threads": 1,
        "n_gpu_layers": 0,
        "verbose": False,
    }
