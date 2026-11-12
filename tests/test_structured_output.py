"""JSON Schema constraints: from the catalog job down to each backend."""

import asyncio
import json

import httpx2

from app import catalog
from app.backends import BackendError
from app.backends.fallback import FallbackBackend
from app.backends.llamacpp import LlamaCppBackend
from app.backends.openai_compatible import OpenAICompatibleBackend
from app.completions import Generation
from app.config import Settings
from app.jobs.models import CATALOG_URL, Job, JobMessage
from app.jobs.processor import BackendRunner, JobProcessor
from tests.fakes import FakeGrammar, InMemoryJobStore

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
CHAT = [{"role": "user", "content": "give me a"}]
# The fake model has no chat template, so llama.cpp is driven with a raw
# prompt; the grammar applies to either.
PROMPT = "give me a"


def test_llama_cpp_compiles_the_schema_into_a_grammar_once():
    backend = LlamaCppBackend(Settings().model_copy(update={"warmup": False}))
    compiled_before = len(FakeGrammar.compiled)

    backend.generate(PROMPT, 32, 0.0, 1.0, None, json_schema=SCHEMA)
    backend.generate(PROMPT, 32, 0.0, 1.0, None, json_schema=SCHEMA)

    first, second = backend._model.calls
    assert json.loads(first["grammar"].schema) == SCHEMA
    # Compiled on first use, reused afterwards.
    assert second["grammar"] is first["grammar"]
    assert len(FakeGrammar.compiled) == compiled_before + 1


def test_llama_cpp_without_a_schema_samples_freely():
    backend = LlamaCppBackend(Settings().model_copy(update={"warmup": False}))

    backend.generate(PROMPT, 32, 0.0, 1.0, None)

    assert "grammar" not in backend._model.calls[0]


def _remote(handler) -> OpenAICompatibleBackend:
    settings = Settings(
        backend="openai_compatible",
        remote_base_url="http://upstream.test/v1",
        model_name="m",
    )
    return OpenAICompatibleBackend(settings, transport=httpx2.MockTransport(handler))


def _reply(request):
    body = {
        "choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    return httpx2.Response(200, json=body)


def test_openai_compatible_requests_strict_structured_output():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return _reply(request)

    _remote(handler).generate(CHAT, 32, 0.0, 1.0, None, json_schema=SCHEMA)

    assert sent[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "output", "schema": SCHEMA, "strict": True},
    }


def test_openai_compatible_without_a_schema_sends_no_response_format():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return _reply(request)

    _remote(handler).generate(CHAT, 32, 0.0, 1.0, None)

    assert "response_format" not in sent[0]


class Recording:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.kwargs: list[dict] = []

    def generate(self, *args, **kwargs):
        self.kwargs.append(kwargs)
        if self.fail:
            raise BackendError("down", status_code=503)
        return {
            "text": "{}",
            "finish_reason": "stop",
            "tokens_prompt": 1,
            "tokens_completion": 1,
        }

    def close(self) -> None:
        pass


def test_fallback_passes_the_schema_to_whichever_backend_serves():
    primary, fallback = Recording(fail=True), Recording()
    backend = FallbackBackend(
        primary, fallback, "fallback-model", failure_threshold=3, cooldown_seconds=30
    )

    backend.generate(CHAT, 32, 0.0, 1.0, None, None, json_schema=SCHEMA)

    assert primary.kwargs == [{"json_schema": SCHEMA}]
    assert fallback.kwargs == [{"json_schema": SCHEMA}]


def test_the_runner_hands_the_generation_schema_to_the_backend():
    backend = Recording()
    runner = BackendRunner(backend, timeout_seconds=5, concurrency=1)
    generation = Generation(CHAT, 32, 0.0, 1.0, None, json_schema=SCHEMA)

    asyncio.run(runner.run(generation))
    runner.close()

    assert backend.kwargs == [{"json_schema": SCHEMA}]


def test_catalog_jobs_ask_for_the_product_schema():
    store = InMemoryJobStore()
    job_id = "job_" + "d" * 32
    store.jobs[job_id] = Job(id=job_id, url=CATALOG_URL, status="queued")
    seen = []

    class Runner:
        async def run(self, generation):
            seen.append(generation)
            raise RuntimeError("stop here")

        def close(self):
            pass

    message = JobMessage(job_id=job_id, url=CATALOG_URL, body={"title": "Chair"})
    worker = JobProcessor(
        store, Runner(), model_name="m", max_attempts=3, timeout_seconds=5
    )
    asyncio.run(worker.process(message.model_dump_json().encode()))

    assert seen[0].json_schema is catalog.SCHEMA
