#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES All rights reserved.
#
# genai_perf.sh — wrapper for genai-perf profile, mirroring aiperf.sh.
#
# Called as:
#   bash genai_perf.sh --result-dir <dir> --model <model> --port <port> \
#                      [--report-name <name>] -- <genai-perf-profile-args>...
#
# Context flags (before --):
#   --result-dir   Directory where artifacts and the final report are written.
#   --model        HuggingFace model identifier (e.g. Qwen/Qwen3-0.6B).
#   --port         HTTP port the dynamo.frontend is listening on.
#   --report-name  Output CSV name (default: benchmark_report.csv).
#
# Everything after -- is passed directly to "genai-perf profile".

set -Eeuo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── genai-perf + perf_analyzer: prefer cloudaix .venv ────────────────────
# genai-perf invokes perf_analyzer as a subprocess via PATH lookup, so both
# binaries must be reachable. Prepending the venv bin covers both.
_VENV_BIN="${_SCRIPT_DIR}/../../.venv/bin"
if [[ -x "${_VENV_BIN}/genai-perf" ]]; then
  export PATH="${_VENV_BIN}:${PATH}"
  GENAI_PERF="${_VENV_BIN}/genai-perf"
else
  GENAI_PERF="genai-perf"
fi

result_dir=""
model=""
port=8000
report_name="benchmark_report.csv"
cmd=""                       # override launch command (empty = use resolved $GENAI_PERF binary)
genai_perf_profile_args=()

log() {
  echo "[$(date '+%F %T') $(hostname)]: $*"
}

_require_value() {
  local flag="$1" val="${2-}"
  if [[ -z "$val" || "$val" == --* ]]; then
    log "ERROR: $flag requires a value (got: '${val:-<empty>}')" >&2; exit 1
  fi
}

# Collect all flags after -- into genai_perf_profile_args.
# Kept as a plain array (not associative) so bare boolean flags like --streaming
# are preserved correctly without an eval round-trip.
parse_genai_perf_args() {
  genai_perf_profile_args=("$@")
}

# Parse context flags (before --); delegate everything after -- to
# parse_genai_perf_args. Mirrors the process_args / parse_genai_perf_args
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
      --)            shift; parse_genai_perf_args "$@"; break ;;
      --*)           echo "[$(date '+%F %T') $(hostname)]: WARNING: ignoring unknown context flag: $1" >&2; shift 1 ;;
      *)             shift ;;
    esac
  done

  log "Parsed args:
    result_dir:  $result_dir
    model:       $model
    port:        $port
    report_name: $report_name
    cmd:         ${cmd:-<default>}
    profile_args: ${genai_perf_profile_args[*]:-}"
}

process_results() {
  local artifact_dir="$result_dir/genai_perf_artifacts"
  local csv_path
  # genai-perf >=0.0.16 writes profile_export_genai_perf.csv inside a run-specific subdir;
  # older versions write profile_genai_perf.csv directly in artifact_dir.
  # Search recursively for either pattern.
  csv_path=$(find "$artifact_dir" -name "*genai_perf*.csv" -print -quit 2>/dev/null || true)
  if [[ -n "$csv_path" ]]; then
    cp "$csv_path" "$result_dir/$report_name"
    log "genai-perf report saved to $result_dir/$report_name"
  else
    log "ERROR: no *genai_perf*.csv found in $artifact_dir — genai-perf benchmark may not have completed"
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
  # Build the launch command: use custom cmd if provided, else resolved $GENAI_PERF binary.
  local -a run_cmd
  if [[ -n "$cmd" ]]; then
    read -ra run_cmd <<< "$cmd"
    run_cmd+=(profile)
  else
    if [[ ! -x "$GENAI_PERF" ]] && ! command -v "$GENAI_PERF" > /dev/null 2>&1; then
      log "ERROR: genai-perf binary not found or not executable: $GENAI_PERF"; exit 1
    fi
    run_cmd=("$GENAI_PERF" profile)
  fi

  local artifact_dir="$result_dir/genai_perf_artifacts"
  # Remove stale artifacts from any previous run so process_results only
  # finds CSV files produced by this invocation.
  rm -rf "$artifact_dir"
  log "Launching genai-perf (cmd=${run_cmd[*]}, model=$model, url=localhost:${port})"

  "${run_cmd[@]}" \
    --model        "$model" \
    --url          "localhost:${port}" \
    --artifact-dir "$artifact_dir" \
    "${genai_perf_profile_args[@]}"

  log "genai-perf run complete"
  process_results
}

main "$@"
exit 0
