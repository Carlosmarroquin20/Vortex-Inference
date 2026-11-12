# 0003. Canary releases with Argo Rollouts and Prometheus analysis

- Status: accepted
- Date: 2026-11-10

## Context

ADR 0002 promotes each image to staging and prod by commit, and a rolling
update replaces the pods. A rolling update only checks readiness. A release
that starts, passes `/health/ready` and then fails half its completions, or
answers twice as slowly, rolls out to every replica. It stays there until
someone notices and reverts. For an inference service, "the pod is ready"
and "the release is good" are different claims: the model loads, the
probes pass, and a prompt-templating bug still breaks every chat request.

The goal is a release that is exposed to real traffic in steps. Each step is
judged on that traffic, and the release is withdrawn automatically when the
numbers are worse than what the service normally does. "Normally" has to be
measured, not guessed.

## Decision

**The API is an Argo Rollouts `Rollout` in the GitOps environments.** The
shared component (`k8s/components/gitops-environment`) turns the base
Deployment into a Rollout with a JSON patch (`rollout.yaml`). The pod
template, probes and resources stay exactly as the base and the component
define them. `make deploy` and the base keep the Deployment. KEDA's
ScaledObject targets the Rollout (`argoproj.io/v1alpha1`, kind `Rollout`)
through its scale subresource. The job workers stay a Deployment, because
asynchronous jobs are retried and dead-lettered rather than served live.

**Steps:**

1. 10%;
2. analysis;
3. 50%;
4. analysis;
5. 100%.

A failed analysis aborts the rollout. If the canary is not ready within 10
minutes (`progressDeadlineAbort`), the rollout aborts too.

**Canary by replica count first, with no traffic router.** The Service keeps
selecting stable and canary pods alike, so the canary's share of traffic is
its share of ready pods:

- `setWeight: 10` on prod's three replicas becomes one canary pod next to
  three stable ones, about 25% of requests, not 10%;
- `setWeight: 50` becomes two next to two.

This is coarser than a traffic router, but it needs nothing beyond the
Service. Precise weights (an exact 10%, header-based routing, mirroring)
would come from adding `trafficRouting` with the Gateway API plugin or an
ingress controller. That changes the Rollout and not the analysis.

**Analysis on the canary only.** The `vortex-canary` AnalysisTemplate queries
Prometheus, and every query selects `rollouts_pod_template_hash`. The
ServiceMonitor copies that hash from the pod label Argo Rollouts puts on
each revision's pods. The stable pods' traffic cannot dilute a bad canary,
and a canary cannot pass on the stable pods' history. The metrics are new
HTTP-level series (`vortex_http_responses_total`,
`vortex_http_response_duration_seconds`), recorded by the request
middleware for the API routes. They see what a client sees, including
failures that never reach the engine.

Each check uses a 90s window, measured three times 30s apart, after 90s of
traffic:

| Check | Fails when |
|-------|-----------|
| request-volume | fewer than `min-requests` completions reached the canary, in two of three windows |
| error-rate | the 5xx share is above `max-error-rate` |
| p95-latency | the p95 is above `max-p95-seconds` in two of three windows |

The volume check is what makes "no data" a failure instead of a pass.
Without it, a canary that received no requests would show a 0% error rate
and an empty latency, and would be promoted.

**Thresholds from a measured baseline.** `scripts/canary_baseline.py` drives
the stable release with the analysis load. That load is `scripts/loadgen.py`,
the existing load generator, run in-cluster by `scripts/traffic.sh` with
fixed parameters. The script evaluates the AnalysisTemplate's own queries
against the stable pods every 30 seconds. The run is in
`evals/baselines/canary.json`. The rules:

- `min-requests`: half of what the canary's single pod would receive in the
  quietest window. While a canary runs, the Service has one pod more than
  the baseline had (`maxSurge: 1`), so the canary's share is the window total
  divided by ready pods + 1. It is halved because kube-proxy does not split
  short windows evenly.
- `max-error-rate`: twice the worst window's 5xx share, but never below one
  failed request in a minimum-volume window. A healthy baseline with zero
  errors would otherwise give a threshold of zero.
- `max-p95-seconds`: 1.5 times the worst window's p95, rounded up to 0.1s.
  The histogram's buckets are 0.5s apart in this range, so a tighter margin
  would fail on interpolation noise.

The measured baseline (2026-11-12, CI run 43261634727) used prod running
`sha-111a835`, three ready pods, 2 concurrent clients, 16-token
completions and 420 s of load:

| Measurement | Worst window (10 windows) | Threshold |
|-------------|---------------------------|-----------|
| Completions per window, all pods | 265.9 (66.5 per pod among four) | `min-requests` = 33 |
| 5xx share | 0 (0 of 1,309 requests) | `max-error-rate` = 0.03 (1/33) |
| p95, server histogram | 1.32 s | `max-p95-seconds` = 2.0 |

The client saw a p95 of 1.05 s for the same requests (`loadgen.py`). The
server figure is higher because `histogram_quantile` interpolates linearly
inside the 1.0–1.5 s bucket. The threshold is compared with the server
figure, so it was derived from that figure too.

The load is closed-loop with two clients, so throughput (about 3.1 requests
per second) does not depend on the number of pods. At 10%, the canary pod is
one of four and gets about 66 completions per window. At 50%, it gets about
133.

**The first derivation was wrong, and a run showed it.** The thresholds were
first derived per pod among the baseline's three pods: `min-requests` 44,
`max-error-rate` 0.023. That overlooked the surge pod. A later CI run then
aborted a good release. The canary received 33.7 completions in its first
window, on a runner that ran about twice as slowly as the baseline's (p95
1.83 s) and while the pod was still cold. Two corrections came out of it,
both applied to the same measured windows:

- the derivation now divides by pods + 1;
- the volume check tolerates one short window out of three, as the latency
  check already did.

"No data" still cannot pass, because zero traffic fails every window.

The thresholds hold only for that load and that model. A different load,
model or node size needs a new baseline. The script is the procedure.

**A bad release for the demonstration.** `VORTEX_FAULT_RATE` (default 0, set
in no profile) makes the API answer that fraction of completions with a
500, before inference. `scripts/demo-canary.sh` commits releases to a
branch the cluster tracks:

- a good release, which must reach 100%;
- the same release with `VORTEX_FAULT_RATE=0.5`, which the first analysis
  must stop;
- a `git revert` of that commit, which must return the Rollout to Healthy.

CI runs it and fails if any step ends otherwise. The GIF in the README is
that run's transcript, replayed as a time-lapse with VHS.

## Alternatives considered

- **Flagger.** It covers the same ground, but it creates and owns shadow
  Deployments next to the one in git. That fits less naturally with Argo
  CD, which would show those objects as drift. Argo Rollouts is from the
  same project as Argo CD, which shows the canary in the Application view.
- **Traffic-routed canary from the start (Gateway API, NGINX, Istio).** It
  gives exact weights, but it needs a router installed and configured
  before the first release, plus Gateway resources per environment. The
  replica-based canary proves the analysis and the abort path with what the
  cluster already has. The router is a separate, additive step.
- **Blue-green with a pre-promotion analysis.** This is simpler to reason
  about, but the new release gets no production traffic before it takes all
  of it. The analysis would then judge synthetic traffic only, which is
  what the smoke test already does.
- **Judging on the inference metrics (`vortex_inference_requests_total`,
  `vortex_inference_duration_seconds`).** These describe the engine. They
  miss failures before it (validation, overload rejections, a broken route
  or middleware), and they exclude queue time. Clients see the HTTP
  response, and that is what the release has to get right.
- **Comparing canary against stable (a Kayenta-style relative check).**
  This is better in principle for noisy traffic, because both see the same
  load at the same time. However, with one to three pods per side, the
  comparison has too few samples to be stable. Fixed thresholds from a
  baseline are easier to explain and to audit.
- **Thresholds chosen by hand ("1% errors, 2s p95").** These look
  reasonable and are unrelated to what this model on this hardware
  actually does. A threshold below the normal p95 fails every release, and
  one far above it never catches a regression.

## Consequences

- A release reaches 100% of prod only after two passing analyses on its own
  traffic, about six minutes. A failing one is withdrawn within about three
  minutes of its first canary pod becoming ready, without anyone running a
  command.
- The canary percentages are approximate, because they are set by pod
  counts. The analysis does not depend on them, since it reads the canary's
  own series.
- An aborted rollout leaves the Application `Degraded` until git changes.
  `git revert` of the release commit is the rollback, consistent with
  ADR 0002: the cluster never ends up running something git does not
  describe.
- An environment with too little traffic cannot promote. In real use the
  minimum volume, or the windows, would be tuned to the environment's
  traffic. Staging would get synthetic load during a release.
