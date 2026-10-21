#!/usr/bin/env python3
"""
Measures llama.cpp prompt processing and generation speed for one GGUF model.

Runs inside the inference image, against llama-cpp-python directly rather
than through the API, so the numbers isolate the engine from HTTP, queueing
and sampling. Each configuration is loaded in turn and measured on the same
synthetic prompt:

    prefill  the prompt evaluated in one call: prompt tokens per second, what
             dominates time to first token
    decode   tokens evaluated one at a time after it: what a streaming
             client sees as time per output token

Run it under the same CPU limit and thread caps as the pod, or the thread
settings compare nothing useful. The image does not contain scripts/, so it
is mounted:

    docker run --rm --cpus 2 -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \\
        -v ./models:/models:ro -v ./scripts:/bench:ro \\
        --entrypoint python vortex-inference:1.0.0 /bench/bench_llama.py \\
        --model /models/phi-3-mini-4k-instruct-q4_k_m.gguf \\
        --threads 2 --threads-batch 2,0 --batch 512,16

--threads-batch 0 reproduces llama-cpp-python's default, one thread per CPU
the process can see. A --batch below 32 keeps the BLAS backend out of prompt
processing, since ggml offloads only larger matrix products to it.
"""

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass

# Filler text for the synthetic prompt; its content does not affect speed.
_FILLER = (
    "The lighthouse keeper climbed the stairs at dusk, checked the lamp, wound "
    "the clockwork and wrote the weather into the log before the long night. "
)


@dataclass(frozen=True)
class Result:
    threads: int
    threads_batch: int
    batch: int
    load_seconds: float
    prompt_tokens: int
    prefill_tokens_per_second: float
    decode_tokens_per_second: float


def build_prompt(llm, tokens: int) -> list[int]:
    """Exactly `tokens` prompt tokens, BOS included."""
    text = _FILLER
    while len(llm.tokenize(text.encode("utf-8"))) < tokens:
        text += _FILLER
    return llm.tokenize(text.encode("utf-8"))[:tokens]


def measure(model: str, threads: int, threads_batch: int, batch: int, args) -> Result:
    from llama_cpp import Llama

    started = time.perf_counter()
    llm = Llama(
        model_path=model,
        n_ctx=args.prompt_tokens + args.gen_tokens + 16,
        n_threads=threads,
        # None is llama-cpp-python's default: os.cpu_count() threads.
        n_threads_batch=threads_batch or None,
        n_batch=batch,
        n_ubatch=batch,
        verbose=False,
    )
    load_seconds = time.perf_counter() - started
    prompt = build_prompt(llm, args.prompt_tokens)

    prefill, decode = [], []
    # The first pass also pays for faulting the memory-mapped weights in and
    # is not counted.
    for attempt in range(args.repeat + 1):
        llm.reset()
        started = time.perf_counter()
        llm.eval(prompt)
        prefill_seconds = time.perf_counter() - started

        # Decoding cost does not depend on which token is fed back, so the
        # last prompt token stands in for sampling.
        started = time.perf_counter()
        for _ in range(args.gen_tokens):
            llm.eval([prompt[-1]])
        decode_seconds = time.perf_counter() - started

        if attempt > 0:
            prefill.append(len(prompt) / prefill_seconds)
            decode.append(args.gen_tokens / decode_seconds)

    return Result(
        threads=threads,
        threads_batch=threads_batch or os.cpu_count(),
        batch=batch,
        load_seconds=round(load_seconds, 2),
        prompt_tokens=len(prompt),
        prefill_tokens_per_second=round(statistics.median(prefill), 1),
        decode_tokens_per_second=round(statistics.median(decode), 1),
    )


def format_table(results: list[Result]) -> str:
    header = (
        f"{'threads':>7} {'batch thr':>9} {'n_batch':>7} "
        f"{'prefill tok/s':>13} {'decode tok/s':>12}"
    )
    rows = [
        f"{r.threads:>7} {r.threads_batch:>9} {r.batch:>7} "
        f"{r.prefill_tokens_per_second:>13} {r.decode_tokens_per_second:>12}"
        for r in results
    ]
    return "\n".join([header, *rows])


def _ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",")]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=_ints, default=[2], help="comma-separated")
    parser.add_argument(
        "--threads-batch",
        type=_ints,
        default=[0],
        help="comma-separated; 0 is llama-cpp-python's default (all CPUs)",
    )
    parser.add_argument("--batch", type=_ints, default=[512], help="comma-separated")
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--gen-tokens", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json", help="also write the results to this file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        f"cpus visible: {os.cpu_count()}, "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}",
        file=sys.stderr,
    )
    results = []
    for threads in args.threads:
        for threads_batch in args.threads_batch:
            for batch in args.batch:
                result = measure(args.model, threads, threads_batch, batch, args)
                print(json.dumps(asdict(result)), file=sys.stderr)
                results.append(result)
    print(format_table(results))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump([asdict(r) for r in results], handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
