#!/usr/bin/env python3
"""
Runs an evaluation dataset against the chat completions endpoint.

Each case in the dataset (JSON Lines, see evals/datasets/) is a conversation
and a list of deterministic checks on the reply. Replies are generated with
temperature 0, so a run measures the model and backend behind the server,
not sampling noise, and the same dataset compares a local GGUF model with a
remote provider on equal terms. Check types:

    contains       value appears (case-insensitive)
    contains_any   at least one of values appears (case-insensitive)
    not_contains   value does not appear (case-insensitive)
    regex          pattern matches (re.search; ignore_case optional)
    json           the reply, without a Markdown code fence, parses as JSON;
                   with keys, it is an object holding all of them
    max_words      at most value words
    line_count     between min and max non-empty lines

The report gives a pass rate overall and per tag, and every failed check
with the reply that failed it. With --min-pass-rate the exit status is 1
when the run falls below it, which makes the script usable as a release gate:

    python scripts/run_evals.py --url http://127.0.0.1:8080 \\
        --dataset evals/datasets/core.jsonl --min-pass-rate 0.8 --json report.json

Only the Python standard library is used, as in scripts/loadgen.py.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field


@dataclass
class CaseResult:
    id: str
    tags: list[str]
    passed: bool
    reply: str
    failures: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    completion_tokens: int = 0
    error: str | None = None


def load_dataset(path: str) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            for required in ("id", "messages", "checks"):
                if required not in case:
                    raise ValueError(f"{path}:{number}: case without {required!r}")
            for check in case["checks"]:
                if check.get("type") not in CHECKS:
                    raise ValueError(
                        f"{path}:{number}: unknown check type {check.get('type')!r}"
                    )
            cases.append(case)
    ids = [case["id"] for case in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"{path}: duplicate case ids {sorted(duplicates)}")
    return cases


def _strip_fence(text: str) -> str:
    # Models often wrap JSON in a Markdown code block even when asked not to;
    # the check is about the JSON, not the wrapping.
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.DOTALL)
    return match.group(1) if match else text.strip()


def _contains(reply, check):
    return check["value"].lower() in reply.lower()


def _contains_any(reply, check):
    return any(value.lower() in reply.lower() for value in check["values"])


def _not_contains(reply, check):
    return check["value"].lower() not in reply.lower()


def _regex(reply, check):
    flags = re.IGNORECASE if check.get("ignore_case") else 0
    return re.search(check["pattern"], reply.strip(), flags) is not None


def _json(reply, check):
    try:
        value = json.loads(_strip_fence(reply))
    except json.JSONDecodeError:
        return False
    keys = check.get("keys")
    return keys is None or (isinstance(value, dict) and all(k in value for k in keys))


def _max_words(reply, check):
    return len(reply.split()) <= check["value"]


def _line_count(reply, check):
    lines = [line for line in reply.splitlines() if line.strip()]
    return check.get("min", 0) <= len(lines) <= check.get("max", len(lines))


CHECKS = {
    "contains": _contains,
    "contains_any": _contains_any,
    "not_contains": _not_contains,
    "regex": _regex,
    "json": _json,
    "max_words": _max_words,
    "line_count": _line_count,
}


def describe(check: dict) -> str:
    details = {k: v for k, v in check.items() if k != "type"}
    return f"{check['type']} {json.dumps(details, ensure_ascii=False)}"


def grade(case: dict, reply: str) -> list[str]:
    """The checks the reply fails, described; empty when it passes."""
    return [describe(c) for c in case["checks"] if not CHECKS[c["type"]](reply, c)]


def complete(url: str, case: dict, args) -> tuple[str, int]:
    body = {
        "messages": case["messages"],
        "max_tokens": case.get("max_tokens", args.max_tokens),
        "temperature": 0,
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(args.api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # The scheme is validated in parse_args(); only http(s) reaches here.
    request = urllib.request.Request(  # noqa: S310
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:  # noqa: S310
        payload = json.loads(response.read())
    reply = payload["choices"][0]["message"]["content"] or ""
    return reply, payload.get("usage", {}).get("completion_tokens", 0)


def run_case(url: str, case: dict, args) -> CaseResult:
    started = time.perf_counter()
    try:
        reply, tokens = complete(url, case, args)
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
        detail = (
            exc.read().decode("utf-8", "replace")[:200] if hasattr(exc, "read") else ""
        )
        return CaseResult(
            id=case["id"],
            tags=case.get("tags", []),
            passed=False,
            reply="",
            latency_seconds=round(time.perf_counter() - started, 3),
            error=f"{type(exc).__name__}: {exc} {detail}".strip(),
        )
    failures = grade(case, reply)
    return CaseResult(
        id=case["id"],
        tags=case.get("tags", []),
        passed=not failures,
        reply=reply,
        failures=failures,
        latency_seconds=round(time.perf_counter() - started, 3),
        completion_tokens=tokens,
    )


def summarize(results: list[CaseResult]) -> dict:
    by_tag: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        for tag in result.tags:
            by_tag[tag].append(result.passed)
    passed = sum(r.passed for r in results)
    latencies = sorted(r.latency_seconds for r in results)
    return {
        "cases": len(results),
        "passed": passed,
        "errors": sum(r.error is not None for r in results),
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "by_tag": {
            tag: round(sum(outcomes) / len(outcomes), 4)
            for tag, outcomes in sorted(by_tag.items())
        },
        "latency_p50_seconds": latencies[len(latencies) // 2] if latencies else 0.0,
        "completion_tokens": sum(r.completion_tokens for r in results),
    }


def format_report(summary: dict, results: list[CaseResult]) -> str:
    lines = [
        f"passed {summary['passed']}/{summary['cases']} "
        f"({summary['pass_rate']:.0%}), errors {summary['errors']}, "
        f"latency p50 {summary['latency_p50_seconds']}s",
        "by tag: "
        + ", ".join(f"{tag} {rate:.0%}" for tag, rate in summary["by_tag"].items()),
    ]
    for result in results:
        if result.passed:
            continue
        lines.append(f"FAIL {result.id}")
        if result.error:
            lines.append(f"     error: {result.error}")
        for failure in result.failures:
            lines.append(f"     {failure}")
        if result.reply:
            lines.append(f"     reply: {result.reply[:160]!r}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", required=True, help="base URL of the server")
    parser.add_argument("--dataset", default="evals/datasets/core.jsonl")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0, help="per case, s")
    parser.add_argument(
        "--api-key-env",
        default="VORTEX_API_KEY",
        help="environment variable holding the bearer token, if any",
    )
    parser.add_argument("--min-pass-rate", type=float, help="exit 1 below this")
    parser.add_argument("--json", dest="json_path", help="write the report here")
    args = parser.parse_args(argv)
    if not args.url.startswith(("http://", "https://")):
        parser.error("--url must start with http:// or https://")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_dataset(args.dataset)
    # One case at a time: latency is reported per case, and a run should
    # not load the server more than one client would.
    results = [run_case(args.url, case, args) for case in cases]
    summary = summarize(results)
    print(format_report(summary, results))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset": args.dataset,
                    "summary": summary,
                    "cases": [asdict(r) for r in results],
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
    if args.min_pass_rate is not None and summary["pass_rate"] < args.min_pass_rate:
        print(
            f"pass rate {summary['pass_rate']:.0%} is below the required "
            f"{args.min_pass_rate:.0%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
