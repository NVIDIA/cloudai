#!/usr/bin/env bash
set -Eeuo pipefail

log() { echo "[$(date +%F\ %T) $(hostname)]: $*"; }

: "${FRONTEND_URL:?FRONTEND_URL is not set}"
: "${AIPERF_MODEL:=model}"
: "${AIPERF_ENDPOINT:=v1/chat/completions}"
: "${AIPERF_FAILURE_MARKER:=/cloudai_run_results/failure-marker.txt}"

rm -rf /cloudai_run_results/aiperf_artifacts/round_1
mkdir -p /cloudai_run_results/aiperf_artifacts/round_1
log 'Running round_1: aiperf profile --model model --endpoint-type chat --streaming --url "$FRONTEND_URL" --concurrency 1 --request-count 50 --synthetic-input-tokens-mean 300 --output-tokens-mean 500 --artifact-dir /cloudai_run_results/aiperf_artifacts/round_1 --no-server-metrics'
phase_status=0
set +e
aiperf profile --model model --endpoint-type chat --streaming --url "$FRONTEND_URL" --concurrency 1 --request-count 50 --synthetic-input-tokens-mean 300 --output-tokens-mean 500 --artifact-dir /cloudai_run_results/aiperf_artifacts/round_1 --no-server-metrics > /cloudai_run_results/aiperf_round_1.log 2>&1
phase_status=$?
set -e
if [[ "$phase_status" -ne 0 ]]; then
  log 'AIPerf phase round_1 failed'
  exit "$phase_status"
fi
if [[ "$phase_status" -eq 0 ]]; then
  mkdir -p /cloudai_run_results
  cp /cloudai_run_results/aiperf_artifacts/round_1/profile_export_aiperf.csv /cloudai_run_results/aiperf_round_1_report.csv
  log 'AIPerf report saved to /cloudai_run_results/aiperf_round_1_report.csv'
  if [[ -f "$AIPERF_FAILURE_MARKER" ]]; then
    log 'FATAL: failure marker found between AIPerf phases'
    exit 1
  fi
  if ! curl -fsS -X POST "${FRONTEND_URL}/${AIPERF_ENDPOINT}" -H 'Content-Type: application/json' -d "{\"model\":\"${AIPERF_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"stream\":false,\"max_tokens\":1}" >/dev/null; then
    log 'FATAL: frontend health probe failed between AIPerf phases'
    exit 1
  fi
fi

rm -rf /cloudai_run_results/aiperf_artifacts/round_2
mkdir -p /cloudai_run_results/aiperf_artifacts/round_2
log 'Running round_2: aiperf profile --model model --endpoint-type chat --streaming --url "$FRONTEND_URL" --concurrency 2 --request-count 10 --synthetic-input-tokens-mean 300 --output-tokens-mean 500 --artifact-dir /cloudai_run_results/aiperf_artifacts/round_2 --no-server-metrics'
phase_status=0
set +e
aiperf profile --model model --endpoint-type chat --streaming --url "$FRONTEND_URL" --concurrency 2 --request-count 10 --synthetic-input-tokens-mean 300 --output-tokens-mean 500 --artifact-dir /cloudai_run_results/aiperf_artifacts/round_2 --no-server-metrics > /cloudai_run_results/aiperf_round_2.log 2>&1
phase_status=$?
set -e
if [[ "$phase_status" -ne 0 ]]; then
  log 'AIPerf phase round_2 failed'
  exit "$phase_status"
fi
if [[ "$phase_status" -eq 0 ]]; then
  mkdir -p /cloudai_run_results
  cp /cloudai_run_results/aiperf_artifacts/round_2/profile_export_aiperf.csv /cloudai_run_results/aiperf_round_2_report.csv
  log 'AIPerf report saved to /cloudai_run_results/aiperf_round_2_report.csv'
  mkdir -p /cloudai_run_results
  cp /cloudai_run_results/aiperf_round_2_report.csv /cloudai_run_results/aiperf_report.csv
  log 'Final AIPerf report saved to /cloudai_run_results/aiperf_report.csv'
fi
