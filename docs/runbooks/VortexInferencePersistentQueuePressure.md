# VortexInferencePersistentQueuePressure

**Meaning.** More than 5 requests have been active across all pods for 10
minutes. KEDA should have scaled out well before this.

**Impact.** Rising latency and, eventually, timeouts.

**Diagnosis.**
1. `kubectl describe scaledobject vortex-inference-scaler -n $NS`:
   check the `Ready` and `Active` conditions and recent events.
2. `kubectl get hpa -n $NS`: compare current and desired replicas.
   Desired at the maximum means capacity is exhausted, not that KEDA is broken.
3. If desired replicas are higher than current, the new pods are blocked.
   Check for Pending pods and quota rejections:
   `kubectl get events -n $NS | grep -i quota`.
4. If KEDA reports errors querying Prometheus, verify the server address in
   `k8s/base/keda-autoscaler.yaml` matches the Prometheus service.

**Mitigation.** Restore KEDA's metric source, free node capacity, or raise
`maxReplicaCount` together with the ResourceQuota (see `k8s/base/namespace.yaml`
for the sizing formula).
