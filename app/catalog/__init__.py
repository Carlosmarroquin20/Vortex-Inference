"""
Catalog enrichment: a messy product title and description in, a structured
record out, validated against product.schema.json (ADR 0006).

Runs as an asynchronous job (POST /v1/jobs with url /v1/catalog/enrichments)
on the same queue, workers and retry topology as completions. The model is
asked for a JSON object only; its answer is parsed and validated here, never
trusted. An answer that does not validate is a failed attempt: the job is
retried with the validation error in the prompt, so the next attempt can
correct it, and after the last attempt the message is dead-lettered for
inspection.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field

from app.backends.base import ChatMessage

SCHEMA: dict[str, Any] = json.loads(
    Path(__file__).with_name("product.schema.json").read_text(encoding="utf-8")
)
CATEGORIES: list[str] = SCHEMA["properties"]["category"]["enum"]
_VALIDATOR = Draft202012Validator(SCHEMA)

# Status of an attempt whose answer did not match the schema: the request
# was understood, the content produced was not acceptable.
INVALID_OUTPUT_STATUS = 422


class CatalogEnrichmentRequest(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=5000)
    # Room for the JSON object; a longer answer is not a better one.
    max_tokens: int = Field(default=256, ge=64, le=1024)

    model_config = {"extra": "forbid"}


class InvalidOutput(ValueError):
    """The model's answer is not a JSON object matching the schema."""


_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    """Strips markup the model gains nothing from reading.

    HTML entities are decoded (twice: feeds often escape them twice) and
    tags become line breaks, so the description arrives as plain text and
    fewer prompt tokens are spent on markup.
    """
    text = html.unescape(html.unescape(text))
    text = _TAG.sub("\n", text)
    lines = (_SPACE.sub(" ", line).strip(" |;") for line in text.splitlines())
    return "\n".join(line for line in lines if line)


_SYSTEM = f"""You extract product data for a catalog. Answer with one JSON object and nothing else.

Fields:
- category: one of {", ".join(CATEGORIES)}
- brand: the brand name only, without a retailer prefix such as "Amazon Brand -"
- material: the main material, as the listing states it
- dimensions: {{"length": number, "width": number, "height": number, "unit": "in" | "cm" | "mm" | "m" | "ft"}}, as stated in the listing
- title: the product name cleaned up: brand and key attributes, normal capitalisation, no codes, promotions or symbols

Example answer:
{{"category": "lamp", "brand": "Rivet", "material": "Metal", "dimensions": {{"length": 10.0, "width": 10.0, "height": 22.5, "unit": "in"}}, "title": "Rivet Modern Arc Table Lamp with Linen Shade, Brass"}}"""  # noqa: E501


def messages(
    request: CatalogEnrichmentRequest, previous_error: str | None
) -> list[ChatMessage]:
    """The conversation for one attempt.

    A retry repeats the request with the reason the previous answer was
    rejected; with greedy decoding the same prompt would produce the same
    invalid answer, so the feedback is what makes a retry worth its cost.
    """
    title, description = clean_text(request.title), clean_text(request.description)
    user = f"Title: {title}\nDescription:\n{description}"
    if previous_error:
        user += (
            f"\n\nYour previous answer was rejected: {previous_error}\n"
            "Answer again with one valid JSON object."
        )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def parse(text: str) -> dict[str, Any]:
    """The validated product record in the model's answer.

    Tolerates what models wrap JSON in (code fences, a sentence before or
    after) by reading from the first "{" to the last "}"; everything inside
    must then be one valid object matching the schema.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise InvalidOutput("the answer contains no JSON object")
    try:
        product = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise InvalidOutput(f"the answer is not valid JSON ({exc.msg})") from exc
    errors = sorted(
        _VALIDATOR.iter_errors(product), key=lambda e: list(e.absolute_path)
    )
    if errors:
        reasons = [
            f"{'/'.join(map(str, error.absolute_path)) or 'object'}: {error.message}"
            for error in errors[:3]
        ]
        raise InvalidOutput("; ".join(reasons)[:500])
    return product
