# VortexInferenceCriticalErrorRate

**Meaning.** More than 5% of requests failed over 3 minutes.

**Impact.** The service is degraded for a significant fraction of traffic.

**Diagnosis and mitigation.** As for [HighErrorRate](VortexInferenceHighErrorRate.md),
with rollback as the first action if the onset coincides with a deployment.
