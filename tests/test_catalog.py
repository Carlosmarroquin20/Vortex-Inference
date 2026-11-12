"""Catalog enrichment: prompt, answer validation, and the worker's decisions."""

import asyncio
import json

import pytest

from app import catalog
from app.completions import Finished
from app.jobs.models import CATALOG_URL, Job, JobError, JobMessage
from app.jobs.processor import JobProcessor
from tests.fakes import InMemoryJobStore
from tests.helpers import metric

JOB_ID = "job_" + "c" * 32
PRODUCT = {
    "category": "chair",
    "brand": "Rivet",
    "material": "Wood",
    "dimensions": {"length": 20.5, "width": 22.0, "height": 31.0, "unit": "in"},
    "title": "Rivet Mid-Century Modern Dining Chair, Walnut",
}
REQUEST = {
    "title": "RIVET MID-CENTURY DINING CHAIR **NEW** #A1B2C3D4",
    "description": "<p>Material: Wood<br>Dimensions: 20.5 x 22.0 x 31.0 in</p>",
}


# ---------------------------------------------------------------------------
# Answer validation
# ---------------------------------------------------------------------------


def test_a_valid_answer_is_returned_as_the_product():
    assert catalog.parse(json.dumps(PRODUCT)) == PRODUCT


def test_code_fences_and_surrounding_text_are_tolerated():
    text = f"Here is the product:\n```json\n{json.dumps(PRODUCT)}\n```\nDone."

    assert catalog.parse(text) == PRODUCT


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ("I cannot help with that.", "no JSON object"),
        ('{"category": "chair",', "no JSON object"),
        ('{"category": "chair", }', "not valid JSON"),
    ],
)
def test_answers_without_a_readable_object_are_rejected(answer, reason):
    with pytest.raises(catalog.InvalidOutput, match=reason):
        catalog.parse(answer)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"category": "spaceship"}, "category: 'spaceship' is not one of"),
        ({"brand": ""}, "brand"),
        (
            {"dimensions": {**PRODUCT["dimensions"], "unit": "furlong"}},
            "dimensions/unit",
        ),
        ({"dimensions": {**PRODUCT["dimensions"], "height": 0}}, "dimensions/height"),
        ({"price": 10}, "Additional properties are not allowed"),
    ],
)
def test_answers_that_break_the_schema_are_rejected_with_the_field(change, reason):
    with pytest.raises(catalog.InvalidOutput, match=reason):
        catalog.parse(json.dumps({**PRODUCT, **change}))


def test_a_missing_field_is_named():
    incomplete = {key: value for key, value in PRODUCT.items() if key != "material"}

    with pytest.raises(
        catalog.InvalidOutput, match="'material' is a required property"
    ):
        catalog.parse(json.dumps(incomplete))


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_markup_is_stripped_and_entities_decoded():
    text = "<p>Made of oak &amp;amp; steel</p><br>  Size:   20 x 30 in | "

    assert catalog.clean_text(text) == "Made of oak & steel\nSize: 20 x 30 in"


def test_the_prompt_lists_the_taxonomy_and_carries_the_input():
    request = catalog.CatalogEnrichmentRequest(**REQUEST)

    system, user = catalog.messages(request, previous_error=None)

    assert all(category in system["content"] for category in catalog.CATEGORIES)
    assert "Dimensions: 20.5 x 22.0 x 31.0 in" in user["content"]
    assert "<p>" not in user["content"]
    assert "rejected" not in user["content"]


def test_a_retry_tells_the_model_why_its_answer_was_rejected():
    request = catalog.CatalogEnrichmentRequest(**REQUEST)

    _, user = catalog.messages(
        request, previous_error="category: 'stool' is not one of"
    )

    assert (
        "Your previous answer was rejected: category: 'stool' is not one of"
        in user["content"]
    )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class AnsweringRunner:
    """Returns the queued answers in order, recording each prompt."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list = []

    async def run(self, generation):
        self.prompts.append(generation)
        text = self.answers.pop(0)
        return Finished(
            result={
                "text": text,
                "finish_reason": "stop",
                "tokens_prompt": 300,
                "tokens_completion": 80,
            },
            duration=2.0,
            tokens_per_second=40.0,
        )

    def close(self) -> None:
        pass


def process(store, runner, max_attempts=3):
    message = JobMessage(job_id=JOB_ID, url=CATALOG_URL, body=REQUEST)
    worker = JobProcessor(
        store,
        runner,
        model_name="test-model",
        max_attempts=max_attempts,
        timeout_seconds=5,
    )
    return asyncio.run(worker.process(message.model_dump_json().encode()))


def queued(store, **fields):
    store.jobs[JOB_ID] = Job(id=JOB_ID, url=CATALOG_URL, status="queued", **fields)
    return store


def test_a_valid_answer_is_stored_as_the_enriched_product():
    store, runner = queued(InMemoryJobStore()), AnsweringRunner(json.dumps(PRODUCT))
    valid_before = metric("vortex_catalog_outputs_total", result="valid")
    enriched_before = metric("vortex_catalog_products_total", outcome="enriched")

    decision = process(store, runner)

    job = store.jobs[JOB_ID]
    assert (decision.action, decision.outcome) == ("ack", "succeeded")
    assert job.status == "succeeded"
    assert job.result == {
        "object": "catalog.enrichment",
        "model": "test-model",
        "product": PRODUCT,
        "attempts": 1,
    }
    assert job.usage.model_dump() == {
        "prompt_tokens": 300,
        "completion_tokens": 80,
        "generation_seconds": 2.0,
    }
    # Greedy decoding: the same input gives the same answer.
    assert runner.prompts[0].temperature == 0.0
    assert metric("vortex_catalog_outputs_total", result="valid") == valid_before + 1
    assert (
        metric("vortex_catalog_products_total", outcome="enriched")
        == enriched_before + 1
    )


def test_an_invalid_answer_is_retried_with_its_reason():
    store = queued(InMemoryJobStore())
    answer = json.dumps({**PRODUCT, "category": "stool"})
    invalid_before = metric("vortex_catalog_outputs_total", result="invalid")

    decision = process(store, AnsweringRunner(answer))

    job = store.jobs[JOB_ID]
    assert (decision.action, decision.outcome) == ("retry", "retried")
    assert job.status == "queued"
    assert job.error.status_code == 422
    assert "category: 'stool' is not one of" in job.error.message
    # The rejected attempt's tokens count towards the product's cost.
    assert job.usage.completion_tokens == 80
    assert (
        metric("vortex_catalog_outputs_total", result="invalid") == invalid_before + 1
    )


def test_the_retry_prompt_carries_the_rejection_and_usage_accumulates():
    rejection = JobError(
        status_code=422, message="category: 'stool' is not one of [...]"
    )
    store = queued(
        InMemoryJobStore(),
        attempts=1,
        error=rejection,
        usage={
            "prompt_tokens": 300,
            "completion_tokens": 80,
            "generation_seconds": 2.0,
        },
    )
    runner = AnsweringRunner(json.dumps(PRODUCT))

    decision = process(store, runner)

    job = store.jobs[JOB_ID]
    assert decision.outcome == "succeeded"
    assert "Your previous answer was rejected" in runner.prompts[0].prompt[1]["content"]
    assert job.result["attempts"] == 2
    assert job.usage.model_dump() == {
        "prompt_tokens": 600,
        "completion_tokens": 160,
        "generation_seconds": 4.0,
    }


def test_a_retry_after_a_backend_failure_gets_the_plain_prompt():
    store = queued(
        InMemoryJobStore(), attempts=1, error=JobError(status_code=503, message="busy")
    )
    runner = AnsweringRunner(json.dumps(PRODUCT))

    process(store, runner)

    assert "rejected" not in runner.prompts[0].prompt[1]["content"]


def test_no_valid_answer_on_the_last_attempt_fails_and_dead_letters():
    store = queued(InMemoryJobStore(), attempts=2)
    failed_before = metric("vortex_catalog_products_total", outcome="failed")

    decision = process(store, AnsweringRunner("not json at all"), max_attempts=3)

    job = store.jobs[JOB_ID]
    assert (decision.action, decision.outcome) == ("reject", "failed")
    assert job.status == "failed"
    assert job.error.status_code == 422
    assert (
        metric("vortex_catalog_products_total", outcome="failed") == failed_before + 1
    )


def test_an_invalid_body_fails_without_running():
    store = queued(InMemoryJobStore())
    message = JobMessage(
        job_id=JOB_ID, url=CATALOG_URL, body={"description": "no title"}
    )
    worker = JobProcessor(
        store, AnsweringRunner(), model_name="m", max_attempts=3, timeout_seconds=5
    )

    decision = asyncio.run(worker.process(message.model_dump_json().encode()))

    assert decision.outcome == "failed"
    assert store.jobs[JOB_ID].error.status_code == 400
