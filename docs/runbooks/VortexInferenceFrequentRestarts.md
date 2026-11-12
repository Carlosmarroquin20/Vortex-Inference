# VortexInferenceFrequentRestarts

**Meaning.** A container restarted more than twice in 15 minutes.

**Impact.** Each restart drops the pod's queue and costs 60-90s of model load.

**Diagnosis.**
1. `kubectl describe pod <pod> -n $NS`: the last state's reason
   (`OOMKilled`, `Error`) and the events (liveness probe failures).
2. `kubectl logs <pod> -n $NS --previous` for the crashed
   container.
3. Events such as `Liveness probe failed: HTTP probe failed with statuscode: 503`
   mean `/health/live` reported a stalled worker: a llama.cpp call ignored its
   stop signal for more than 60s past the deadline (the probe body says
   `Inference worker is stalled past its deadline.`). The restart is the intended recovery; recurring stalls point to a
   specific request shape. Find the request ID of the last request on that
   pod before the restart and reproduce it.

**Mitigation.** For OOM, see [PodHighMemoryUsage](VortexInferencePodHighMemoryUsage.md).
For model load failures, see [NoReadyPods](VortexInferenceNoReadyPods.md).
