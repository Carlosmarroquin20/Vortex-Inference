#!/usr/bin/env python3
"""
One-line status of the API's Argo Rollouts canary, for scripts and logs.

  python3 scripts/rollout_status.py NAMESPACE           summary line
  python3 scripts/rollout_status.py NAMESPACE phase     Healthy, Progressing, Degraded
  python3 scripts/rollout_status.py NAMESPACE stable    stable revision's pod hash
  python3 scripts/rollout_status.py NAMESPACE message   the controller's status message

The summary names the canary step, the ready pods of each revision, and the
latest measurement of each metric in the canary's current AnalysisRun, e.g.

  Progressing | step 2/4 analysis | stable 3, canary 1 | Running:
  request-volume=31(Successful) error-rate=0(Successful) p95-latency=0.9(Successful)

Reads the cluster through kubectl, without the Argo Rollouts kubectl plugin.
"""

from __future__ import annotations

import json
import subprocess
import sys


def kubectl_json(*args: str) -> dict:
    out = subprocess.run(
        ["kubectl", *args, "-o", "json"], check=True, capture_output=True, text=True
    ).stdout
    return json.loads(out)


def ready(pod: dict) -> bool:
    if pod["metadata"].get("deletionTimestamp"):
        return False
    return any(
        c["type"] == "Ready" and c["status"] == "True"
        for c in pod["status"].get("conditions", [])
    )


def summary(namespace: str, rollout: dict) -> str:
    status = rollout["status"]
    steps = rollout["spec"]["strategy"]["canary"]["steps"]
    index = status.get("currentStepIndex", len(steps))
    if index >= len(steps):
        where = "all steps done"
    elif "setWeight" in steps[index]:
        where = f"step {index + 1}/{len(steps)} setWeight {steps[index]['setWeight']}%"
    else:
        where = f"step {index + 1}/{len(steps)} {next(iter(steps[index]))}"

    stable, canary = status.get("stableRS"), status.get("currentPodHash")
    pods = kubectl_json(
        "get", "pods", "-n", namespace, "-l", "app.kubernetes.io/name=vortex-inference"
    )["items"]
    hashes = [
        p["metadata"]["labels"].get("rollouts-pod-template-hash")
        for p in pods
        if ready(p)
    ]
    pod_text = f"stable {hashes.count(stable)}"
    if canary != stable:
        pod_text += f", canary {hashes.count(canary)}"

    measured = ""
    if canary != stable:
        runs = [
            run
            for run in kubectl_json("get", "analysisruns", "-n", namespace)["items"]
            if run["metadata"]["labels"].get("rollouts-pod-template-hash") == canary
        ]
        if runs:
            latest = max(runs, key=lambda run: run["metadata"]["creationTimestamp"])
            parts = []
            for metric in latest.get("status", {}).get("metricResults", []):
                if metric.get("measurements"):
                    value = metric["measurements"][-1].get("value", "?").strip("[]")
                    parts.append(f"{metric['name']}={value}({metric['phase']})")
            if parts:
                phase = latest.get("status", {}).get("phase", "")
                measured = f" | {phase}: " + " ".join(parts)

    return f"{status.get('phase', '?')} | {where} | {pod_text}{measured}"


def main() -> int:
    namespace = sys.argv[1]
    field = sys.argv[2] if len(sys.argv) > 2 else "summary"
    rollout = kubectl_json("get", "rollout", "-n", namespace, "vortex-inference")
    status = rollout.get("status", {})
    if field == "phase":
        print(status.get("phase", ""))
    elif field == "stable":
        print(status.get("stableRS", ""))
    elif field == "message":
        print(status.get("message", ""))
    else:
        print(summary(namespace, rollout))
    return 0


if __name__ == "__main__":
    sys.exit(main())
