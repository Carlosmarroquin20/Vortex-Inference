#!/usr/bin/env python3
"""
Client-side load generator for Vortex's OpenAI-compatible endpoints.

Measures what a client experiences, which server metrics cannot: end-to-end
latency including the network and the ingress, and for streamed requests the
time to first token and the time per output token as chunks actually arrive.
Results are printed as a summary and can be written as JSON for comparing
runs (backends, models, concurrency levels).

Closed loop: CONCURRENCY workers each send their next request as soon as the
previous one finishes, so the offered load adapts to the server's speed. That
measures latency at a given concurrency, not at a fixed arrival rate.

Standard library only, so it runs wherever Python 3.10+ does.

Examples:
    VORTEX_API_KEY=... python3 scripts/loadgen.py --url http://10.96.0.10 \\
        --endpoint chat --stream --concurrency 4 --requests 40
    python3 scripts/loadgen.py --url http://127.0.0.1:8080 --json run.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

# Varied prompts so consecutive requests do not reuse a cached prefix, which
# would make prompt processing look faster than it is.
DEFAULT_PROMPTS = [
    "Explain the CAP theorem in distributed systems in two sentences.",
    "What is the difference between a process and a thread? Be concise.",
    "Describe what a Kubernetes operator is in one paragraph.",
    "What problem does consistent hashing solve in distributed caches?",
    "Explain gradient descent in machine learning briefly.",
    "What is the role of etcd in Kubernetes? Be specific.",
    "Describe the producer-consumer problem and one solution.",
    "What is the difference between TCP and UDP? Answer concisely.",
]


@dataclass
class Result:
    index: int
    ok: bool
    status: int  # HTTP status; 0 when no response was received
    latency_s: float
    ttft_s: float | None = None
    tpot_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


@dataclass
class Config:
    url: str
    endpoint: str
    stream: bool
    concurrency: int
    requests: int
    max_tokens: int
    temperature: float
    timeout_s: float
    prompts: list[str] = field(repr=False)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def build_body(config: Config, prompt: str) -> dict:
    body: dict = {
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "stream": config.stream,
    }
    if config.endpoint == "chat":
        body["messages"] = [{"role": "user", "content": prompt}]
    else:
        body["prompt"] = prompt
    if config.stream:
        # Usage arrives in a final chunk only when asked for.
        body["stream_options"] = {"include_usage": True}
    return body


def chunk_text(chunk: dict, endpoint: str) -> str:
    """Text carried by one streamed chunk, empty for role or usage chunks."""
    for choice in chunk.get("choices") or []:
        if endpoint == "chat":
            return (choice.get("delta") or {}).get("content") or ""
        return choice.get("text") or ""
    return ""


def read_stream(response, endpoint: str, started: float, result: Result) -> None:
    """Consumes an SSE body, timing chunks as they arrive."""
    first_text_at = last_text_at = None
    text_chunks = 0
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if "error" in chunk:
            # The server's in-stream error event: the request failed after
            # the response was committed with 200.
            result.ok = False
            result.error = f"stream error: {chunk['error'].get('message')}"
            return
        if chunk.get("usage"):
            result.prompt_tokens = chunk["usage"].get("prompt_tokens")
            result.completion_tokens = chunk["usage"].get("completion_tokens")
        if chunk_text(chunk, endpoint):
            now = time.perf_counter()
            first_text_at = first_text_at or now
            last_text_at = now
            text_chunks += 1

    if first_text_at is not None:
        result.ttft_s = first_text_at - started
        tokens = result.completion_tokens or text_chunks
        if tokens >= 2:
            result.tpot_s = (last_text_at - first_text_at) / (tokens - 1)
    if result.completion_tokens is None:
        result.completion_tokens = text_chunks


def send(config: Config, api_key: str | None, index: int) -> Result:
    prompt = config.prompts[index % len(config.prompts)]
    path = "/v1/chat/completions" if config.endpoint == "chat" else "/v1/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # The scheme is validated in parse_args(); only http(s) reaches here.
    request = urllib.request.Request(  # noqa: S310
        config.url.rstrip("/") + path,
        data=json.dumps(build_body(config, prompt)).encode(),
        headers=headers,
        method="POST",
    )

    started = time.perf_counter()
    result = Result(index=index, ok=True, status=0, latency_s=0.0)
    try:
        # The scheme is validated in parse_args(); only http(s) reaches here.
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:  # noqa: S310
            result.status = response.status
            if config.stream:
                read_stream(response, config.endpoint, started, result)
            else:
                payload = json.loads(response.read())
                usage = payload.get("usage") or {}
                result.prompt_tokens = usage.get("prompt_tokens")
                result.completion_tokens = usage.get("completion_tokens")
    except urllib.error.HTTPError as exc:
        result.ok, result.status = False, exc.code
        result.error = exc.read().decode("utf-8", errors="replace")[:200]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"[:200]
    result.latency_s = time.perf_counter() - started
    return result


def run(
    config: Config, api_key: str | None, progress=None
) -> tuple[list[Result], float]:
    counter = iter(range(config.requests))
    lock = threading.Lock()
    results: list[Result] = []

    def worker() -> None:
        while True:
            with lock:
                index = next(counter, None)
            if index is None:
                return
            result = send(config, api_key, index)
            with lock:
                results.append(result)
            if progress is not None:
                progress(result, len(results), config.requests)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        for _ in range(config.concurrency):
            pool.submit(worker)
    wall = time.perf_counter() - started
    return sorted(results, key=lambda r: r.index), wall


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in 0..100), as numpy's default."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def summarize(results: list[Result], wall_s: float) -> dict:
    ok = [r for r in results if r.ok]
    failures: dict[str, int] = {}
    for r in results:
        if not r.ok:
            key = str(r.status) if r.status else "no response"
            failures[key] = failures.get(key, 0) + 1
    output_tokens = sum(r.completion_tokens or 0 for r in ok)
    return {
        "requests": len(results),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        "failures_by_status": failures,
        "wall_seconds": wall_s,
        "requests_per_second": len(ok) / wall_s if wall_s > 0 else 0.0,
        "output_tokens_per_second": output_tokens / wall_s if wall_s > 0 else 0.0,
        "latency_seconds": distribution([r.latency_s for r in ok]),
        "ttft_seconds": distribution([r.ttft_s for r in ok if r.ttft_s is not None]),
        "tpot_seconds": distribution([r.tpot_s for r in ok if r.tpot_s is not None]),
    }


def format_summary(summary: dict) -> str:
    lines = [
        f"requests   {summary['succeeded']}/{summary['requests']} succeeded"
        + (f", failures {summary['failures_by_status']}" if summary["failed"] else ""),
        f"wall       {summary['wall_seconds']:.2f}s, "
        f"{summary['requests_per_second']:.2f} req/s, "
        f"{summary['output_tokens_per_second']:.1f} output tokens/s",
        "",
        f"{'':12}{'mean':>9}{'p50':>9}{'p90':>9}{'p95':>9}{'p99':>9}{'max':>9}",
    ]
    rows = [
        ("latency", summary["latency_seconds"]),
        ("ttft", summary["ttft_seconds"]),
        ("tpot", summary["tpot_seconds"]),
    ]
    for name, dist in rows:
        if dist is None:
            continue
        lines.append(
            f"{name + ' (s)':12}"
            + "".join(
                f"{dist[k]:>9.3f}" for k in ("mean", "p50", "p90", "p95", "p99", "max")
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> tuple[Config, str | None, str | None]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True, help="base URL, e.g. http://10.96.0.10")
    parser.add_argument("--endpoint", choices=["completions", "chat"], default="chat")
    parser.add_argument("--stream", action="store_true", help="stream responses (SSE)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=180.0, help="per request, s")
    parser.add_argument(
        "--prompts", help="file with one prompt per line (default: built-in set)"
    )
    parser.add_argument(
        "--api-key-env",
        default="VORTEX_API_KEY",
        help="environment variable holding the API key; read from the environment "
        "so the key never appears in the process list",
    )
    parser.add_argument("--json", dest="json_path", help="write results to this file")
    args = parser.parse_args(argv)

    if not args.url.startswith(("http://", "https://")):
        parser.error("--url must start with http:// or https://")
    if args.concurrency < 1 or args.requests < 1:
        parser.error("--concurrency and --requests must be at least 1")

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as fh:
            prompts = [line.strip() for line in fh if line.strip()]
        if not prompts:
            parser.error(f"{args.prompts} contains no prompts")

    config = Config(
        url=args.url,
        endpoint=args.endpoint,
        stream=args.stream,
        concurrency=args.concurrency,
        requests=args.requests,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout,
        prompts=prompts,
    )
    return config, os.environ.get(args.api_key_env), args.json_path


def main(argv: list[str] | None = None) -> int:
    config, api_key, json_path = parse_args(argv)

    def progress(result: Result, done: int, total: int) -> None:
        status = "ok" if result.ok else f"FAILED {result.status or ''} {result.error}"
        print(f"  [{done}/{total}] {result.latency_s:7.2f}s {status}", file=sys.stderr)

    mode = "streamed" if config.stream else "buffered"
    print(
        f"{config.requests} {mode} {config.endpoint} requests, concurrency "
        f"{config.concurrency}, max_tokens {config.max_tokens} -> {config.url}",
        file=sys.stderr,
    )
    results, wall = run(config, api_key, progress)
    summary = summarize(results, wall)
    print(format_summary(summary))

    if json_path:
        report = {
            "config": {k: v for k, v in asdict(config).items() if k != "prompts"},
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"results written to {json_path}", file=sys.stderr)

    # Nonzero only when nothing worked, which points at the setup (URL, key,
    # network) rather than at the load; partial failures are the measurement.
    return 0 if summary["succeeded"] else 1


if __name__ == "__main__":
    sys.exit(main())
