#!/usr/bin/env bash
set -Eeuo pipefail

log() { echo "[$(date +%F\ %T) $(hostname)]: $*"; }

: "${FRONTEND_URL:?FRONTEND_URL is not set}"

rm -rf /cloudai_run_results/aiperf_artifacts/round_1
mkdir -p /cloudai_run_results/aiperf_artifacts/round_1
log 'Running round_1: aiperf profile --model model --endpoint-type chat --streaming --artifact-dir /cloudai_run_results/aiperf_artifacts/round_1 --no-server-metrics --concurrency 1 --request-count 50 --synthetic-input-tokens-mean 300 --output-tokens-mean 500 --url "$FRONTEND_URL"'
aiperf profile --model model --endpoint-type chat --streaming --artifact-dir /cloudai_run_results/aiperf_artifacts/round_1 --no-server-metrics --concurrency 1 --request-count 50 --synthetic-input-tokens-mean 300 --output-tokens-mean 500 --url "$FRONTEND_URL" > /cloudai_run_results/aiperf_round_1.log 2>&1
mkdir -p /cloudai_run_results
cp /cloudai_run_results/aiperf_artifacts/round_1/profile_export_aiperf.csv /cloudai_run_results/aiperf_round_1_report.csv
log 'AIPerf report saved to /cloudai_run_results/aiperf_round_1_report.csv'

rm -rf /cloudai_run_results/aiperf_artifacts/round_2
mkdir -p /cloudai_run_results/aiperf_artifacts/round_2
log 'Running round_2: aiperf profile --model model --endpoint-type chat --streaming --artifact-dir /cloudai_run_results/aiperf_artifacts/round_2 --no-server-metrics --concurrency 2 --request-count 10 --synthetic-input-tokens-mean 300 --output-tokens-mean 500 --url "$FRONTEND_URL"'
aiperf profile --model model --endpoint-type chat --streaming --artifact-dir /cloudai_run_results/aiperf_artifacts/round_2 --no-server-metrics --concurrency 2 --request-count 10 --synthetic-input-tokens-mean 300 --output-tokens-mean 500 --url "$FRONTEND_URL" > /cloudai_run_results/aiperf_round_2.log 2>&1
mkdir -p /cloudai_run_results
cp /cloudai_run_results/aiperf_artifacts/round_2/profile_export_aiperf.csv /cloudai_run_results/aiperf_round_2_report.csv
log 'AIPerf report saved to /cloudai_run_results/aiperf_round_2_report.csv'
mkdir -p /cloudai_run_results
cp /cloudai_run_results/aiperf_round_2_report.csv /cloudai_run_results/aiperf_report.csv
log 'Final AIPerf report saved to /cloudai_run_results/aiperf_report.csv'
