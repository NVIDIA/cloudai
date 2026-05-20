#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES All rights reserved.
#
# aiperf.sh — wrapper for aiperf profile, mirroring genai_perf.sh.
#
# Called as:
#   bash aiperf.sh --result-dir <dir> --model <model> --port <port> \
#                  [--report-name <name>] -- <aiperf-profile-args>...
#
# Context flags (before --):
#   --result-dir   Directory where artifacts and the final report are written.
#   --model        HuggingFace model identifier (e.g. Qwen/Qwen3-0.6B).
#   --port         HTTP port the dynamo.frontend is listening on.
#   --report-name  Output CSV name (default: benchmark_report.csv).
#
# Everything after -- is passed directly to "aiperf profile".
# This wrapper adds the derived flags: --url, --endpoint-type, --streaming,
# --artifact-dir, --no-server-metrics.

set -Eeuo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── aiperf binary: prefer cloudaix .venv ─────────────────────────────────
_VENV_AIPERF="${_SCRIPT_DIR}/../../.venv/bin/aiperf"
if [[ -x "$_VENV_AIPERF" ]]; then
  AIPERF="$_VENV_AIPERF"
else
  AIPERF="aiperf"
fi

result_dir=""
model=""
port=8000
report_name="benchmark_report.csv"
cmd=""                  # override launch command (empty = use resolved $AIPERF binary)
aiperf_profile_args=()

log() {
  echo "[$(date '+%F %T') $(hostname)]: $*"
}

_require_value() {
  local flag="$1" val="${2-}"
  if [[ -z "$val" || "$val" == -* ]]; then
    log "ERROR: $flag requires a value (got: '${val:-<empty>}')" >&2; exit 1
  fi
}

# Collect all flags after -- into aiperf_profile_args.
# Kept as a plain array so bare boolean flags are preserved correctly.
parse_aiperf_args() {
  aiperf_profile_args=("$@")
}

# Parse context flags (before --); delegate everything after -- to
# parse_aiperf_args. Mirrors the process_args / parse_genai_perf_args
# split in ai_dynamo's genai_perf.sh.
process_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --result-dir)  _require_value "$1" "${2-}"; result_dir="$2";  shift 2 ;;
      --model)       _require_value "$1" "${2-}"; model="$2";       shift 2 ;;
      --port)
        _require_value "$1" "${2-}"
        if [[ ! "$2" =~ ^[0-9]+$ ]]; then
          log "ERROR: --port must be numeric (got: '$2')" >&2; exit 1
        fi
        port="$2"; shift 2 ;;
      --report-name) _require_value "$1" "${2-}"; report_name="$2"; shift 2 ;;
      --cmd)         _require_value "$1" "${2-}"; cmd="$2";         shift 2 ;;
      --)            shift; parse_aiperf_args "$@"; break ;;
      --*)           log "WARNING: ignoring unknown context flag: $1" >&2; shift 1 ;;
      *)             shift ;;
    esac
  done

  log "Parsed args:
    result_dir:  $result_dir
    model:       $model
    port:        $port
    report_name: $report_name
    cmd:         ${cmd:-<default>}
    profile_args: ${aiperf_profile_args[*]:-}"
}

process_results() {
  local artifact_dir="$result_dir/aiperf_artifacts"
  local csv_path
  csv_path=$(find "$artifact_dir" -name "*aiperf*.csv" -print -quit 2>/dev/null || true)
  if [[ -n "$csv_path" ]]; then
    cp "$csv_path" "$result_dir/$report_name"
    log "aiperf report saved to $result_dir/$report_name"
  else
    log "ERROR: no CSV found in $artifact_dir — aiperf benchmark may not have completed"
    exit 1
  fi
}

main() {
  process_args "$@"

  if [[ -z "$result_dir" ]]; then
    log "ERROR: --result-dir is required"; exit 1
  fi
  if [[ -z "$model" ]]; then
    log "ERROR: --model is required"; exit 1
  fi
  # Build the launch command: use custom cmd if provided, else resolved $AIPERF binary.
  local -a run_cmd
  if [[ -n "$cmd" ]]; then
    read -ra run_cmd <<< "$cmd"
    run_cmd+=(profile)
  else
    if [[ ! -x "$AIPERF" ]] && ! command -v "$AIPERF" > /dev/null 2>&1; then
      log "ERROR: aiperf binary not found or not executable: $AIPERF"; exit 1
    fi
    run_cmd=("$AIPERF" profile)
  fi

  local artifact_dir="$result_dir/aiperf_artifacts"
  # Remove stale artifacts from any previous run so process_results only
  # finds CSV files produced by this invocation.
  rm -rf "$artifact_dir"
  log "Launching aiperf (cmd=${run_cmd[*]}, model=$model, url=localhost:${port})"

  "${run_cmd[@]}" \
    --model        "$model" \
    --url          "localhost:${port}" \
    --endpoint-type chat \
    --streaming \
    --artifact-dir "$artifact_dir" \
    --no-server-metrics \
    "${aiperf_profile_args[@]}"

  log "aiperf run complete"
  process_results
}

main "$@"
exit 0
