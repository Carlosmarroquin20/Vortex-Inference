#!/usr/bin/env bash
# =============================================================================
# Vortex Inference — Load Test Script
#
# Purpose: generate sustained concurrent inference traffic to trigger and
# observe KEDA's Prometheus-driven scale-out behavior in real time.
#
# Design rationale: this script intentionally avoids external load testing
# tools (hey, vegeta, k6) to eliminate setup friction. Pure bash + curl
# achieves sufficient concurrency to move the vortex_inference_active_requests
# gauge above the KEDA threshold (3), which is all that is needed to
# demonstrate horizontal autoscaling on a single-node cluster.
#
# Expected behavior:
#   t=0s    1 replica running, active_requests begins climbing
#   t=15s   KEDA polls Prometheus, detects sum(active_requests) > 3
#   t=30s   KEDA patches Deployment spec.replicas = 2
#   t=60s–120s  New pod passes startupProbe and joins the Endpoints slice
#   t=300s  cooldownPeriod expires, KEDA scales back to 1 replica (if idle)
#
# Usage:
#   bash scripts/load-test.sh [CONCURRENCY] [TOTAL_REQUESTS] [MAX_TOKENS]
#   make load-test
# =============================================================================

set -euo pipefail

# --- Color output ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log_info()  { echo -e "${GREEN}[load-test]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[load-test]${NC} $*"; }
log_error() { echo -e "${RED}[load-test]${NC} $*"; }
log_header(){ echo -e "\n${BOLD}${CYAN}$*${NC}\n"; }

# =============================================================================
# Configuration
# =============================================================================

# CONCURRENCY: number of parallel curl processes.
# Must exceed KEDA threshold (3) to trigger scale-out.
# Set conservatively to avoid OOM on a single-replica pod — the scale-out
# event will absorb the load once the second pod becomes ready.
CONCURRENCY=${1:-5}

# TOTAL_REQUESTS: total inference calls to send across all concurrent workers.
TOTAL_REQUESTS=${2:-20}

# MAX_TOKENS: controls request duration. Lower = faster turnaround, higher =
# longer active_requests gauge inflation = stronger KEDA signal.
# 80 tokens ≈ 40–60s per request on Phi-3-mini at 2 CPU threads.
MAX_TOKENS=${3:-80}

NAMESPACE="vortex-inference"
SVCNAME="vortex-inference-lb"

# Test prompts — varied to avoid KV-cache hits that would shorten inference time.
# Shorter active_requests windows produce weaker KEDA signals.
PROMPTS=(
    "Explain the CAP theorem in distributed systems in two sentences:"
    "What is the difference between a process and a thread? Be concise:"
    "Describe what a Kubernetes operator pattern is in one paragraph:"
    "What problem does consistent hashing solve in distributed caches?"
    "Explain gradient descent optimization in machine learning briefly:"
    "What is the role of an etcd cluster in Kubernetes? Be specific:"
    "Describe the producer-consumer problem and one solution:"
    "What is the difference between TCP and UDP? Answer concisely:"
)

# =============================================================================
# Helpers
# =============================================================================

get_inference_ip() {
    kubectl get svc "${SVCNAME}" \
        -n "${NAMESPACE}" \
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo ""
}

get_replica_count() {
    kubectl get deployment vortex-inference \
        -n "${NAMESPACE}" \
        -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0"
}

get_active_requests() {
    # Queries the metric directly from the pod rather than through Prometheus,
    # providing real-time data without waiting for the scrape interval.
    local pod
    pod=$(kubectl get pod -n "${NAMESPACE}" \
        -l app.kubernetes.io/name=vortex-inference \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [ -z "${pod}" ]; then
        echo "N/A"
        return
    fi

    kubectl exec -n "${NAMESPACE}" "${pod}" -- \
        wget -qO- http://localhost:8080/metrics 2>/dev/null \
        | grep "^vortex_inference_active_requests " \
        | awk '{printf "%.0f", $2}' || echo "0"
}

send_request() {
    local endpoint=$1
    local prompt_index=$(( RANDOM % ${#PROMPTS[@]} ))
    local prompt="${PROMPTS[$prompt_index]}"
    local request_id=$2

    local response
    local http_code
    local start_time
    start_time=$(date +%s%N)

    http_code=$(curl -s -o /tmp/vortex_response_${request_id}.json \
        -w "%{http_code}" \
        --max-time 150 \
        -X POST "${endpoint}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"${prompt}\", \"max_tokens\": ${MAX_TOKENS}, \"temperature\": 0.7}" \
        2>/dev/null || echo "000")

    local end_time
    end_time=$(date +%s%N)
    local duration_ms=$(( (end_time - start_time) / 1000000 ))

    if [ "${http_code}" = "200" ]; then
        local tps
        tps=$(python3 -c "
import json, sys
try:
    data = json.load(open('/tmp/vortex_response_${request_id}.json'))
    print(f\"{data.get('tokens_per_second', 0):.1f}\")
except:
    print('N/A')
" 2>/dev/null || echo "N/A")
        echo -e "  ${GREEN}[req-${request_id}]${NC} OK | ${duration_ms}ms | ${tps} tok/s"
    else
        echo -e "  ${RED}[req-${request_id}]${NC} HTTP ${http_code} | ${duration_ms}ms"
    fi

    rm -f "/tmp/vortex_response_${request_id}.json"
    echo "${http_code}"
}

# =============================================================================
# Monitor replica count in the background
# Prints a timestamped line each time the replica count changes, making the
# KEDA scale-out event visible in the terminal without blocking the load test.
# =============================================================================
monitor_replicas() {
    local last_count="-1"
    local start_epoch
    start_epoch=$(date +%s)

    while true; do
        local current
        current=$(get_replica_count)
        local active
        active=$(get_active_requests)
        local elapsed=$(( $(date +%s) - start_epoch ))

        if [ "${current}" != "${last_count}" ]; then
            echo -e "\n  ${CYAN}[monitor t=${elapsed}s]${NC} Replicas: ${last_count} → ${current} | active_requests: ${active}"
            last_count="${current}"
        fi

        sleep 5
    done
}

# =============================================================================
# Pre-flight checks
# =============================================================================

log_header "Vortex Inference — KEDA Scale-Out Load Test"

log_info "Configuration:"
echo "  Concurrency:     ${CONCURRENCY} parallel requests"
echo "  Total requests:  ${TOTAL_REQUESTS}"
echo "  Max tokens:      ${MAX_TOKENS} per request"
echo "  KEDA threshold:  3 active requests → scale-out"
echo ""

# Verify cluster access
if ! kubectl get namespace "${NAMESPACE}" &>/dev/null; then
    log_error "Namespace '${NAMESPACE}' not found. Run 'make deploy' first."
    exit 1
fi

# Resolve LoadBalancer IP
INFERENCE_IP=$(get_inference_ip)
if [ -z "${INFERENCE_IP}" ]; then
    log_error "LoadBalancer IP not assigned for service '${SVCNAME}'."
    log_error "Run 'make tunnel' in a separate terminal and wait ~10s."
    exit 1
fi

ENDPOINT="http://${INFERENCE_IP}"
log_info "Target endpoint: ${ENDPOINT}/v1/completions"

# Verify the pod is healthy before starting the test
log_info "Checking service health..."
health_response=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${ENDPOINT}/health" 2>/dev/null || echo "000")
if [ "${health_response}" != "200" ]; then
    log_error "Health check failed (HTTP ${health_response}). Pod may not be ready."
    log_error "Check: kubectl get pods -n ${NAMESPACE}"
    exit 1
fi
log_info "Health check passed. Starting load test."

# =============================================================================
# Run load test
# =============================================================================

INITIAL_REPLICAS=$(get_replica_count)
log_info "Initial replica count: ${INITIAL_REPLICAS}"
log_warn "Watch for KEDA scale-out in ~15–120s after active_requests exceeds threshold."
echo ""

# Start background replica monitor — killed automatically on script exit
monitor_replicas &
MONITOR_PID=$!
trap "kill ${MONITOR_PID} 2>/dev/null; exit" INT TERM EXIT

# Track results
SUCCESS_COUNT=0
ERROR_COUNT=0
REQUEST_COUNTER=0

# Send requests in batches of CONCURRENCY
# Each batch fires CONCURRENCY curl processes in parallel, waits for all to
# complete, then fires the next batch. This produces a staircase load profile
# rather than a sustained ramp — adequate for demonstrating KEDA reaction time.
while [ "${REQUEST_COUNTER}" -lt "${TOTAL_REQUESTS}" ]; do
    BATCH_SIZE=$(( CONCURRENCY < (TOTAL_REQUESTS - REQUEST_COUNTER) \
        ? CONCURRENCY \
        : (TOTAL_REQUESTS - REQUEST_COUNTER) ))

    log_info "Sending batch of ${BATCH_SIZE} concurrent requests (${REQUEST_COUNTER}/${TOTAL_REQUESTS} done)..."

    PIDS=()
    RESULTS_FILE="/tmp/vortex_batch_results_$$"

    for (( i=0; i<BATCH_SIZE; i++ )); do
        REQUEST_ID=$(( REQUEST_COUNTER + i ))
        (
            result=$(send_request "${ENDPOINT}" "${REQUEST_ID}")
            echo "${result}" >> "${RESULTS_FILE}"
        ) &
        PIDS+=($!)
    done

    # Wait for all parallel requests in this batch to complete
    for pid in "${PIDS[@]}"; do
        wait "${pid}"
    done

    # Tally results
    if [ -f "${RESULTS_FILE}" ]; then
        while IFS= read -r code; do
            if [ "${code}" = "200" ]; then
                (( SUCCESS_COUNT++ )) || true
            else
                (( ERROR_COUNT++ )) || true
            fi
        done < "${RESULTS_FILE}"
        rm -f "${RESULTS_FILE}"
    fi

    REQUEST_COUNTER=$(( REQUEST_COUNTER + BATCH_SIZE ))
done

# =============================================================================
# Summary
# =============================================================================

# Allow monitor one final check
sleep 3
kill "${MONITOR_PID}" 2>/dev/null || true

FINAL_REPLICAS=$(get_replica_count)
TOTAL_SENT=$(( SUCCESS_COUNT + ERROR_COUNT ))
SUCCESS_RATE=0
if [ "${TOTAL_SENT}" -gt 0 ]; then
    SUCCESS_RATE=$(( SUCCESS_COUNT * 100 / TOTAL_SENT ))
fi

log_header "Load Test Summary"
echo "  Requests sent:     ${TOTAL_SENT}"
echo -e "  Successful:        ${GREEN}${SUCCESS_COUNT}${NC}"
echo -e "  Errors:            ${RED}${ERROR_COUNT}${NC}"
echo "  Success rate:      ${SUCCESS_RATE}%"
echo "  Initial replicas:  ${INITIAL_REPLICAS}"
echo "  Final replicas:    ${FINAL_REPLICAS}"
echo ""

if [ "${FINAL_REPLICAS}" -gt "${INITIAL_REPLICAS}" ]; then
    log_info "KEDA scale-out confirmed: ${INITIAL_REPLICAS} → ${FINAL_REPLICAS} replicas."
else
    log_warn "Replica count unchanged. Possible reasons:"
    log_warn "  - KEDA cooldown period active from a previous test (wait 300s)"
    log_warn "  - Requests completed before KEDA polling interval (15s)"
    log_warn "  - Increase CONCURRENCY or MAX_TOKENS for a longer signal window"
fi

echo ""
log_info "Observe metrics:"
echo "  kubectl get hpa -n ${NAMESPACE} -w"
echo "  kubectl describe scaledobject vortex-inference-scaler -n ${NAMESPACE}"
echo ""
