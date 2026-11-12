#!/usr/bin/env bash
# =============================================================================
# Steady client traffic against one environment, from inside the cluster.
#
# Usage:
#   scripts/traffic.sh start NAMESPACE DURATION_SECONDS [CONCURRENCY]
#   scripts/traffic.sh wait  NAMESPACE      # until done; prints the report
#   scripts/traffic.sh stop  NAMESPACE
#
# Runs scripts/loadgen.py (the same load generator `make loadgen` uses) as a
# Job in the vortex-traffic namespace, against the environment's ClusterIP
# Service. From inside the cluster, kube-proxy spreads the requests over
# every ready pod, stable and canary alike, which is what a replica-count
# canary relies on; a kubectl port-forward would pin them all to one pod.
#
# The load is fixed so that runs are comparable: short buffered completions
# (16 tokens, temperature 0), CONCURRENCY clients in a closed loop. The
# canary baseline (scripts/canary-baseline.sh) is measured under exactly
# this load, and the analysis thresholds hold only under it.
#
# The image is the one the environment runs, so no other image is pulled;
# loadgen.py arrives through a ConfigMap. The namespace has no
# NetworkPolicy; the environments accept the API port from any namespace.
# =============================================================================

set -euo pipefail

command="${1:?usage: traffic.sh start|wait|stop NAMESPACE [DURATION] [CONCURRENCY]}"
target="${2:?usage: traffic.sh start|wait|stop NAMESPACE [DURATION] [CONCURRENCY]}"
ns=vortex-traffic
job="traffic-${target}"

case "${command}" in
start)
  duration="${3:?DURATION_SECONDS required}"
  concurrency="${4:-2}"
  image=$(kubectl get rollout,deployment -n "${target}" -l app.kubernetes.io/component=inference-server \
    -o jsonpath='{.items[0].spec.template.spec.containers[0].image}')
  api_key=$(kubectl get secret -n "${target}" vortex-inference-api-key -o jsonpath='{.data.api-key}')

  kubectl create namespace "${ns}" --dry-run=client -o yaml | kubectl apply -f - > /dev/null
  kubectl label namespace "${ns}" --overwrite \
    pod-security.kubernetes.io/enforce=restricted > /dev/null
  kubectl create configmap loadgen -n "${ns}" --from-file=scripts/loadgen.py \
    --dry-run=client -o yaml | kubectl apply -f - > /dev/null
  # The key is copied as the Secret's base64 data, never decoded here.
  printf 'apiVersion: v1\nkind: Secret\nmetadata:\n  name: %s\n  namespace: %s\ndata:\n  api-key: %s\n' \
    "${job}-key" "${ns}" "${api_key}" | kubectl apply -f - > /dev/null
  kubectl delete job -n "${ns}" "${job}" --ignore-not-found --wait > /dev/null

  kubectl apply -f - > /dev/null <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${job}
  namespace: ${ns}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: loadgen
        image: ${image}
        command:
        - python
        - /loadgen/loadgen.py
        - --url=http://vortex-inference.${target}.svc.cluster.local
        - --endpoint=completions
        - --max-tokens=16
        - --temperature=0
        - --timeout=60
        - --concurrency=${concurrency}
        - --duration=${duration}
        - --quiet
        - --json=-
        env:
        - name: VORTEX_API_KEY
          valueFrom:
            secretKeyRef:
              name: ${job}-key
              key: api-key
        resources:
          requests:
            cpu: 50m
            memory: 64Mi
          limits:
            cpu: 500m
            memory: 256Mi
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: ["ALL"]
        volumeMounts:
        - name: loadgen
          mountPath: /loadgen
      volumes:
      - name: loadgen
        configMap:
          name: loadgen
EOF
  echo "[traffic] ${concurrency} clients against ${target} for ${duration}s"
  ;;
wait)
  kubectl wait -n "${ns}" "job/${job}" --for=condition=complete --timeout=3600s > /dev/null \
    || kubectl wait -n "${ns}" "job/${job}" --for=condition=failed --timeout=1s > /dev/null || true
  kubectl logs -n "${ns}" "job/${job}"
  ;;
stop)
  kubectl delete job -n "${ns}" "${job}" --ignore-not-found > /dev/null
  echo "[traffic] stopped traffic against ${target}"
  ;;
*)
  echo "unknown command: ${command}" >&2
  exit 1
  ;;
esac
