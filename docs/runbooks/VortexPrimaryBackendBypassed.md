# VortexPrimaryBackendBypassed

**Meaning.** For 10 minutes the circuit breaker in front of the primary
backend has been open: the primary failed `VORTEX_FALLBACK_FAILURE_THRESHOLD`
times in a row, so requests go straight to the fallback, and each retry after
`VORTEX_FALLBACK_COOLDOWN_SECONDS` has failed again.

**Impact.** Clients are answered, by the fallback's model. That can cost more
(a paid provider) and answer differently; buffered responses name the
fallback model in `model`.

**Diagnosis.**
1. `sum by (reason) (rate(vortex_backend_fallbacks_total[5m]))`: `overloaded`
   is a provider rate limit or capacity problem, `upstream_error` and
   `timeout` an unreachable or failing upstream.
2. The primary's errors are logged with each fallback:
   `make logs | grep "Using the fallback"`, whose `error` field holds the
   upstream message.

**Mitigation.** Fix the primary (quota, credentials, the upstream service).
The breaker closes by itself on the first successful request after a
cooldown; nothing needs restarting.
