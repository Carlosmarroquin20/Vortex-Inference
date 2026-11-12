# VortexInferenceNoReadyPods

**Meaning.** For 2 minutes no inference pod has been Ready, or none exists.
Read from the pods, so it holds for the local Deployment and for the Rollout
in staging and prod.

**Impact.** Full outage: every request fails at the Service.

**Diagnosis.**
1. `kubectl get pods -n $NS` and `kubectl describe pod <pod>`.
2. `CreateContainerConfigError`: a Secret is missing. Locally, recreate it with
   `make api-key`. Under GitOps, check that the SealedSecret was decrypted:
   `kubectl get sealedsecret -n $NS` (a `Synced=False` condition means the
   cluster's controller lacks the sealing key; see ADR 0002).
3. `CrashLoopBackOff`: read the previous container's logs with
   `kubectl logs <pod> -n $NS --previous`. A `Model failed to load.` line with
   a missing path means the model volume is empty or the file name does not
   match `VORTEX_MODEL_PATH`; an OOM kill (`kubectl describe pod`, reason
   `OOMKilled`) means the model and KV cache exceed the memory limit.
4. Pods Running but not Ready: the readiness probe is failing. A worker
   thread stalled past its deadline also fails readiness; see
   [FrequentRestarts](VortexInferenceFrequentRestarts.md).
5. No pods at all under GitOps: `kubectl get rollout vortex-inference -n $NS`
   and the Application in Argo CD; a sync error leaves the Rollout absent.

**Mitigation.** Fix the specific cause above. If a release caused it, roll it
back with `git revert` (see [Rolling back](README.md#rolling-back)).
