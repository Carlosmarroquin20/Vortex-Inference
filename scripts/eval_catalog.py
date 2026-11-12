#!/usr/bin/env python3
"""
Evaluate catalog enrichment against the labelled dataset, through the queue.

Each row of evals/datasets/catalog.jsonl (built by build_catalog_dataset.py
from public ABO listings) is submitted as a POST /v1/jobs catalog enrichment,
exactly as a client would, and the job's result is compared with the row's
labels:

  category     exact
  brand        equal after case folding and dropping punctuation
  material     equal, or one contains the other, after the same normalisation
               ("Solid Wood" for "wood" is right; "Metal" for "wood" is not)
  dimensions   length, width and height each within 2% after conversion to cm
  title        token F1 against the listing's own name of at least 0.8

A product whose job failed (no valid answer after every attempt) counts as
wrong on every field. Reported alongside: the share of products with a valid
record, the share valid on the first attempt, and per product the attempts,
generation time, tokens and an estimated cost.

Cost: a local model has no per-token price, so it is estimated from compute:
generation seconds x CPU cores per worker x the price of a core-hour
(--cpu-hour-price, default 0.0336 USD: a general-purpose cloud vCPU on
demand, an assumption to adjust). For a hosted model, pass its token prices
instead (--input-price/--output-price, USD per million tokens).

Standard library only.

  python3 scripts/eval_catalog.py --url http://127.0.0.1:8081 \\
      --dataset evals/datasets/catalog.jsonl --json catalog-report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

TO_CM = {"cm": 1.0, "mm": 0.1, "m": 100.0, "in": 2.54, "ft": 30.48}
FIELDS = ("category", "brand", "material", "dimensions", "title")


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def tokens(text: str) -> list[str]:
    return normalise(text).split()


def token_f1(predicted: str, expected: str) -> float:
    p, e = tokens(predicted), tokens(expected)
    if not p or not e:
        return 0.0
    common = sum(min(p.count(t), e.count(t)) for t in set(p))
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(e)
    return 2 * precision * recall / (precision + recall)


def dimensions_match(predicted: dict, expected: dict) -> bool:
    try:
        p_scale, e_scale = TO_CM[predicted["unit"]], TO_CM[expected["unit"]]
        return all(
            abs(predicted[a] * p_scale - expected[a] * e_scale)
            <= 0.02 * expected[a] * e_scale
            for a in ("length", "width", "height")
        )
    except (KeyError, TypeError):
        return False


def grade(product: dict | None, expected: dict) -> dict:
    if product is None:
        return {field: False for field in FIELDS} | {"title_f1": 0.0}
    material_p, material_e = (
        normalise(product["material"]),
        normalise(expected["material"]),
    )
    f1 = token_f1(product["title"], expected["title"])
    return {
        "category": product["category"] == expected["category"],
        "brand": normalise(product["brand"]) == normalise(expected["brand"]),
        "material": bool(material_p)
        and (
            material_p == material_e
            or material_e in material_p
            or material_p in material_e
        ),
        "dimensions": dimensions_match(product["dimensions"], expected["dimensions"]),
        "title": f1 >= 0.8,
        "title_f1": round(f1, 3),
    }


def request(
    url: str, api_key: str | None, method: str = "GET", payload: dict | None = None
):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)  # noqa: S310
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310 - operator-given URL
        return json.load(response)


def run_row(base: str, api_key: str | None, row: dict, timeout: float) -> dict:
    submitted = time.monotonic()
    job = request(
        f"{base}/v1/jobs",
        api_key,
        "POST",
        {"url": "/v1/catalog/enrichments", "body": row["input"]},
    )
    deadline = submitted + timeout
    while job["status"] not in ("succeeded", "failed"):
        if time.monotonic() > deadline:
            break
        time.sleep(1.0)
        job = request(f"{base}/v1/jobs/{job['id']}", api_key)
    product = (
        (job.get("result") or {}).get("product")
        if job["status"] == "succeeded"
        else None
    )
    usage = job.get("usage") or {}
    return {
        "id": row["id"],
        "category": row["expected"]["category"],
        "status": job["status"],
        "attempts": job.get("attempts", 0),
        "error": (job.get("error") or {}).get("message"),
        "grades": grade(product, row["expected"]),
        "product": product,
        "generation_seconds": usage.get("generation_seconds", 0.0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "wall_seconds": round(time.monotonic() - submitted, 2),
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * q / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarise(results: list[dict], args, wall: float) -> dict:
    n = len(results)
    valid = [r for r in results if r["status"] == "succeeded"]
    accuracy = {f: sum(r["grades"][f] for r in results) / n for f in FIELDS}
    per_category: dict[str, dict] = defaultdict(lambda: {"n": 0, "all_fields": 0})
    for r in results:
        bucket = per_category[r["category"]]
        bucket["n"] += 1
        bucket["all_fields"] += all(r["grades"][f] for f in FIELDS)
    seconds = [r["generation_seconds"] for r in results]
    prompt = [r["prompt_tokens"] for r in results]
    completion = [r["completion_tokens"] for r in results]
    compute_cost = [
        s * args.cpus_per_worker * args.cpu_hour_price / 3600 for s in seconds
    ]
    token_cost = [
        (p * args.input_price + c * args.output_price) / 1_000_000
        for p, c in zip(prompt, completion, strict=True)
    ]
    return {
        "products": n,
        "valid_rate": round(len(valid) / n, 4),
        "first_attempt_valid_rate": round(
            sum(r["attempts"] == 1 for r in valid) / n, 4
        ),
        "mean_attempts": round(statistics.fmean(r["attempts"] for r in results), 3),
        "accuracy": {f: round(v, 4) for f, v in accuracy.items()},
        "accuracy_mean": round(statistics.fmean(accuracy.values()), 4),
        "all_fields_correct": round(
            sum(all(r["grades"][f] for f in FIELDS) for r in results) / n, 4
        ),
        "title_f1_mean": round(
            statistics.fmean(r["grades"]["title_f1"] for r in results), 4
        ),
        "per_category_all_fields": {
            c: round(v["all_fields"] / v["n"], 3)
            for c, v in sorted(per_category.items())
        },
        "per_product": {
            "generation_seconds_p50": round(percentile(seconds, 50), 3),
            "generation_seconds_p95": round(percentile(seconds, 95), 3),
            "prompt_tokens_mean": round(statistics.fmean(prompt), 1),
            "completion_tokens_mean": round(statistics.fmean(completion), 1),
            "compute_cost_usd_mean": round(statistics.fmean(compute_cost), 7),
            "token_cost_usd_mean": round(statistics.fmean(token_cost), 7),
        },
        "pricing": {
            "cpu_hour_price_usd": args.cpu_hour_price,
            "cpus_per_worker": args.cpus_per_worker,
            "input_price_usd_per_million": args.input_price,
            "output_price_usd_per_million": args.output_price,
        },
        "throughput_products_per_minute": round(n / wall * 60, 2),
        "wall_seconds": round(wall, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--dataset", default="evals/datasets/catalog.jsonl")
    parser.add_argument("--limit", type=int, help="first N rows only")
    parser.add_argument("--concurrency", type=int, default=8, help="jobs in flight")
    parser.add_argument("--timeout", type=float, default=900.0, help="per product, s")
    parser.add_argument("--cpu-hour-price", type=float, default=0.0336)
    parser.add_argument("--cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--input-price", type=float, default=0.0)
    parser.add_argument("--output-price", type=float, default=0.0)
    parser.add_argument("--min-valid-rate", type=float, default=0.0)
    parser.add_argument(
        "--min-accuracy", type=float, default=0.0, help="mean over fields"
    )
    parser.add_argument("--api-key-env", default="VORTEX_API_KEY")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rows = rows[: args.limit] if args.limit else rows
    if not args.url.startswith(("http://", "https://")):
        parser.error("--url must start with http:// or https://")
    api_key = os.environ.get(args.api_key_env)
    base = args.url.rstrip("/")

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(
            pool.map(lambda row: run_row(base, api_key, row, args.timeout), rows)
        )
    summary = summarise(results, args, time.monotonic() - started)

    print(json.dumps(summary, indent=2))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "results": results}, fh, indent=2)

    failures = []
    if summary["valid_rate"] < args.min_valid_rate:
        failures.append(f"valid rate {summary['valid_rate']} < {args.min_valid_rate}")
    if summary["accuracy_mean"] < args.min_accuracy:
        failures.append(
            f"mean accuracy {summary['accuracy_mean']} < {args.min_accuracy}"
        )
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
