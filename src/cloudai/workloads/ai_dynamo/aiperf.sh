#!/usr/bin/env bash
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# aiperf.sh — aiperf profile wrapper for ai_dynamo workloads.
#
# Called from ai_dynamo.sh's launch_workload() with:
#   bash aiperf.sh --result-dir <dir> --model <model> --url <url> --port <port>
#                  [--cmd <cmd>] [--report-name <name>] [--extra-args <args>]
#                  -- <aiperf-args>...
#
# Context flags (before --) that are recognised and used:
#   --result-dir    Directory where artifacts and the final report are written.
#   --model         HuggingFace model identifier (e.g. Qwen/Qwen3-0.6B).
#   --url           Base URL of the dynamo.frontend (e.g. http://node01).
#   --port          HTTP port the dynamo.frontend is listening on.
#   --report-name   Output CSV name (default: aiperf_report.csv).
#   --cmd           Full launch command including subcommand (default: "aiperf profile").
#   --extra-args    Raw string appended verbatim after all other flags.
#
# All unrecognised flags (--install-dir, --gpus-per-node, etc.) are silently
# consumed so this script is forward-compatible with launch_workload additions.
#
# Everything after -- is passed directly to the aiperf profile invocation.

set -Eeuo pipefail

result_dir=""
model=""
url="http://localhost"
port=8000
report_name="aiperf_report.csv"
cmd="aiperf profile"
declare -a extra_args=()
declare -a aiperf_profile_args=()

log() {
  echo "[$(date '+%F %T') $(hostname)]: $*"
}

_parse_aiperf_args() {
  while [[ $# -ge 2 ]]; do
    case "$1" in
      --*) aiperf_profile_args+=("$1" "$2"); shift 2 ;;
      *)   shift ;;
    esac
  done
  # Capture a trailing lone boolean flag if present.
  # Use if/fi — not [[ ]] && — so set -e does not trigger on a false condition.
  if [[ $# -eq 1 && "$1" == --* ]]; then
    aiperf_profile_args+=("$1")
  fi
}

process_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --result-dir)   result_dir="$2";  shift 2 ;;
      --model)        model="$2";       shift 2 ;;
      --url)          url="$2";         shift 2 ;;
      --port)         port="$2";        shift 2 ;;
      --report-name)  report_name="$2"; shift 2 ;;
      --cmd)          cmd="$2";         shift 2 ;;
      --extra-args)   read -ra extra_args <<< "$2"; shift 2 ;;
      --)             shift; _parse_aiperf_args "$@"; break ;;
      --*)            if [[ -n "${2:-}" && "${2}" != -* ]]; then shift 2; else shift 1; fi ;;  # consume unknown flag; shift 2 only if next arg is a value
      *)              shift ;;
    esac
  done

  log "Parsed args:
    result_dir:   $result_dir
    model:        $model
    url:          $url
    port:         $port
    report_name:  $report_name
    cmd:          $cmd
    extra_args:   ${extra_args[*]:-}
    profile_args: ${aiperf_profile_args[*]:-}"
}

process_results() {
  local artifact_dir="$result_dir/aiperf_artifacts"
  local csv_path
  csv_path=$(find "$artifact_dir" -name "*.csv" -print -quit 2>/dev/null || true)
  if [[ -n "$csv_path" ]]; then
    cp "$csv_path" "$result_dir/$report_name"
    log "aiperf report saved to $result_dir/$report_name"
  else
    log "ERROR: no CSV found in $artifact_dir — aiperf may not have completed"
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

  local full_url="${url}:${port}"
  local artifact_dir="$result_dir/aiperf_artifacts"
  rm -rf "$artifact_dir"

  # Split cmd into an array (e.g. "aiperf profile" → ["aiperf", "profile"])
  local -a run_cmd=()
  read -ra run_cmd <<< "$cmd"

  log "Launching aiperf: ${run_cmd[*]} --model $model --url $full_url"

  "${run_cmd[@]}" \
    --model         "$model" \
    --url           "$full_url" \
    --endpoint-type chat \
    --streaming \
    --artifact-dir  "$artifact_dir" \
    --no-server-metrics \
    "${aiperf_profile_args[@]}" \
    "${extra_args[@]}"

  log "aiperf run complete"
  process_results
}

main "$@"
exit 0
