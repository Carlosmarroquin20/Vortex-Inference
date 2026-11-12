"""scripts/eval_catalog.py: how an enriched product is graded against its labels."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app import catalog

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "eval_catalog", ROOT / "scripts" / "eval_catalog.py"
)
evaluation = importlib.util.module_from_spec(_SPEC)
sys.modules["eval_catalog"] = evaluation
_SPEC.loader.exec_module(evaluation)

EXPECTED = {
    "category": "chair",
    "brand": "Stone & Beam",
    "material": "wood",
    "dimensions": {"length": 20.0, "width": 22.0, "height": 31.0, "unit": "in"},
    "title": "Stone & Beam Dining Chair, Walnut",
}


def test_a_perfect_answer_passes_every_field():
    grades = evaluation.grade(dict(EXPECTED), EXPECTED)

    assert all(grades[field] for field in evaluation.FIELDS)
    assert grades["title_f1"] == 1.0


def test_brand_ignores_case_and_punctuation():
    assert (
        evaluation.grade({**EXPECTED, "brand": "stone and beam"}, EXPECTED)["brand"]
        is False
    )
    assert (
        evaluation.grade({**EXPECTED, "brand": "STONE & BEAM"}, EXPECTED)["brand"]
        is True
    )


@pytest.mark.parametrize(
    ("material", "right"), [("Solid Wood", True), ("Metal", False)]
)
def test_material_accepts_a_more_specific_answer(material, right):
    assert (
        evaluation.grade({**EXPECTED, "material": material}, EXPECTED)["material"]
        is right
    )


def test_dimensions_compare_in_centimetres_within_two_percent():
    metric = {"length": 50.8, "width": 55.88, "height": 78.74, "unit": "cm"}
    off = {**EXPECTED["dimensions"], "height": 33.0}

    assert (
        evaluation.grade({**EXPECTED, "dimensions": metric}, EXPECTED)["dimensions"]
        is True
    )
    assert (
        evaluation.grade({**EXPECTED, "dimensions": off}, EXPECTED)["dimensions"]
        is False
    )


def test_title_needs_most_of_the_listing_name():
    close = evaluation.grade(
        {**EXPECTED, "title": "Stone & Beam Walnut Dining Chair"}, EXPECTED
    )
    shouty = evaluation.grade({**EXPECTED, "title": "CHAIR #A1B2C3"}, EXPECTED)

    assert close["title"] is True
    assert shouty["title"] is False


def test_a_failed_job_is_wrong_on_every_field():
    grades = evaluation.grade(None, EXPECTED)

    assert not any(grades[field] for field in evaluation.FIELDS)


def test_every_dataset_label_is_a_valid_product():
    # The labels are what a perfect answer looks like, so they must pass
    # the same schema the worker enforces.
    rows = (
        (ROOT / "evals" / "datasets" / "catalog.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert len(rows) == 200
    for line in rows:
        catalog.parse(json.dumps(json.loads(line)["expected"]))
