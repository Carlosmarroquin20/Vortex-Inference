#!/usr/bin/env python3
"""
Measure the baseline the canary analysis thresholds are derived from.

Drives an environment's current, healthy release with the analysis load
(scripts/traffic.sh: the existing load generator, fixed parameters) and,
every 30 seconds, evaluates the AnalysisTemplate's own queries against the
stable pods. The windows are therefore measured exactly as a canary's will
be: same metrics, same 90s window, same load.

Thresholds (ADR 0003):

  min-requests     half the completions one pod receives per window at its
                   quietest. Per pod, because a canary starts as one pod;
                   halved, because kube-proxy does not split traffic evenly
                   over short windows.
  max-error-rate   twice the worst window's 5xx rate, but never below one
                   failed request in a minimum-volume window: a healthy
                   baseline with zero errors would otherwise yield a
                   threshold of zero, and one transient error would fail a
                   release.
  max-p95-seconds  1.5 times the worst window's p95, rounded up to 0.1s. The
                   latency histogram's buckets are 0.5s apart in this range,
                   so a tighter margin would fail on interpolation noise.

Usage:
  python3 scripts/canary_baseline.py --namespace vortex-prod \\
      --json evals/baselines/canary.json
Needs kubectl on a cluster bootstrapped with `make bootstrap-gitops`.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "k8s" / "components" / "gitops-environment" / "analysis-template.yaml"
PROMETHEUS_PORT = 19090
WARMUP_S = 90  # one full window of traffic before the first measurement
INTERVAL_S = 30


def kubectl(*args: str) -> str:
    return subprocess.run(
        ["kubectl", *args], check=True, capture_output=True, text=True
    ).stdout


def analysis_queries() -> dict[str, str]:
    """The AnalysisTemplate's queries, as committed."""
    template = yaml.safe_load(TEMPLATE.read_text())
    return {
        m["name"]: m["provider"]["prometheus"]["query"]
        for m in template["spec"]["metrics"]
    }


def render(query: str, namespace: str, pod_hash: str) -> str:
    return query.replace("{{args.namespace}}", namespace).replace(
        "{{args.canary-hash}}", pod_hash
    )


def instant(query: str) -> float | None:
    url = f"http://127.0.0.1:{PROMETHEUS_PORT}/api/v1/query?" + urllib.parse.urlencode(
        {"query": query}
    )
    with urllib.request.urlopen(url, timeout=10) as response:
        result = json.load(response)["data"]["result"]
    if not result:
        return None
    value = float(result[0]["value"][1])
    return None if math.isnan(value) else value


def derive(windows: list[dict], pods: int) -> dict:
    per_pod = [w["request-volume"] / pods for w in windows]
    min_requests = max(1, math.floor(min(per_pod) / 2))
    worst_errors = max(w["error-rate"] or 0.0 for w in windows)
    max_error_rate = max(2 * worst_errors, 1 / min_requests)
    worst_p95 = max(w["p95-latency"] for w in windows if w["p95-latency"] is not None)
    return {
        "min-requests": min_requests,
        "max-error-rate": round(max_error_rate, 3),
        "max-p95-seconds": math.ceil(worst_p95 * 1.5 * 10) / 10,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--namespace", default="vortex-prod")
    parser.add_argument("--duration", type=int, default=420, help="seconds of traffic")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--json", dest="json_path", required=True)
    args = parser.parse_args()

    status = json.loads(
        kubectl(
            "get", "rollout", "-n", args.namespace, "vortex-inference", "-o", "json"
        )
    )
    stable_hash = status["status"]["stableRS"]
    pods = status["status"]["readyReplicas"]
    queries = analysis_queries()

    forward = subprocess.Popen(
        [
            "kubectl",
            "port-forward",
            "-n",
            "monitoring",
            "svc/kube-prometheus-stack-prometheus",
            f"{PROMETHEUS_PORT}:9090",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    windows: list[dict] = []
    try:
        subprocess.run(
            [
                "bash",
                "scripts/traffic.sh",
                "start",
                args.namespace,
                str(args.duration),
                str(args.concurrency),
            ],
            check=True,
            cwd=ROOT,
        )
        started = time.monotonic()
        time.sleep(WARMUP_S)
        while time.monotonic() - started < args.duration - INTERVAL_S:
            window = {
                name: instant(render(q, args.namespace, stable_hash))
                for name, q in queries.items()
            }
            window["elapsed_s"] = round(time.monotonic() - started)
            windows.append(window)
            print(f"[baseline] {json.dumps(window)}", file=sys.stderr)
            time.sleep(INTERVAL_S)
        report = subprocess.run(
            ["bash", "scripts/traffic.sh", "wait", args.namespace],
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        forward.terminate()

    loadgen = json.loads(report.strip().splitlines()[-1])
    baseline = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": args.namespace,
        "image": kubectl(
            "get",
            "rollout",
            "-n",
            args.namespace,
            "vortex-inference",
            "-o",
            "jsonpath={.spec.template.spec.containers[0].image}",
        ),
        "ready_pods": pods,
        "load": {
            "generator": "scripts/loadgen.py via scripts/traffic.sh",
            "concurrency": args.concurrency,
            "duration_s": args.duration,
            "endpoint": "completions",
            "max_tokens": 16,
            "temperature": 0,
        },
        "windows": windows,
        "client": loadgen["summary"],
        "thresholds": derive(windows, pods),
    }
    Path(args.json_path).write_text(json.dumps(baseline, indent=2) + "\n")
    print(json.dumps(baseline["thresholds"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
