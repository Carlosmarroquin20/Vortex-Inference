# VortexInferencePodHighMemoryUsage

**Meaning.** A pod's working set has been above 90% of its 4Gi limit for 5
minutes.

**Impact.** The next long-context request may push it over the limit and get
the container OOM-killed, dropping its queue.

**Diagnosis.** Compare the pod's memory with its context settings
(`VORTEX_N_CTX` in the ConfigMap). The KV cache is allocated for the full
context window at startup, so memory usage is mostly fixed by `n_ctx` and the
model size rather than by traffic.

**Mitigation.** Lower `VORTEX_N_CTX`, use a smaller quantization, or raise the
memory limit together with the ResourceQuota and LimitRange in
`k8s/base/namespace.yaml`. Apply with `make deploy`.
