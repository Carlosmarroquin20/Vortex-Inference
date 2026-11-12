#!/usr/bin/env bash
# =============================================================================
# Replays a demo transcript as a time-lapse, for recording.
#
# Usage: scripts/replay-log.sh LOG [TARGET_SECONDS] [LABEL]
#
# LOG is the tab-separated transcript scripts/demo-canary.sh writes
# (seconds since start, line). Lines are printed in order with their real
# gaps scaled so the whole run plays in about TARGET_SECONDS (default 50),
# and the elapsed real time is kept on each line. The first line states the
# speed-up and where the run came from, so the recording never passes for
# real time.
#
# docs/demo/canary.tape records this with VHS.
# =============================================================================

set -euo pipefail

log="${1:?usage: replay-log.sh LOG [TARGET_SECONDS] [LABEL]}"
target="${2:-50}"
label="${3:-}"

total=$(tail -n 1 "${log}" | cut -f1)
[ "${total}" -gt 0 ] || total=1
# Real seconds per replayed second, rounded up.
factor=$(( (total + target - 1) / target ))

printf '\033[2m# scripts/demo-canary.sh, %s of real time replayed %sx faster%s\033[0m\n\n' \
  "$((total / 60))m$((total % 60))s" "${factor}" "${label:+ (${label})}"

previous=0
while IFS=$'\t' read -r at line; do
  gap=$((at - previous))
  previous=${at}
  # Each real second becomes 1/factor; no single pause runs past 2s.
  delay=$(awk -v g="${gap}" -v f="${factor}" 'BEGIN { d = g / f; if (d > 2) d = 2; printf "%.2f", d }')
  sleep "${delay}"
  case "${line}" in
    "=== "*) printf '\033[1;36m%s\033[0m\n' "${line}" ;;
    PASS:*) printf '[%4ds] \033[1;32m%s\033[0m\n' "${at}" "${line}" ;;
    FAIL:*|*Degraded*) printf '[%4ds] \033[1;31m%s\033[0m\n' "${at}" "${line}" ;;
    "") echo ;;
    *) printf '[%4ds] %s\n' "${at}" "${line}" ;;
  esac
done < "${log}"
sleep 3
