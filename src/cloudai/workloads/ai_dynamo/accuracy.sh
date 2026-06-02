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

set -Eeuo pipefail

result_dir=""
model=""
url="http://localhost"
port=8000
endpoint="v1/chat/completions"
entrypoint=""
cli=""
setup_cmd=""
artifact_dir_name="aiperf_accuracy_artifacts"

log() {
  echo "[$(date '+%F %T') $(hostname)]: $*"
}

process_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --result-dir)          result_dir="$2";        shift 2 ;;
      --model)               model="$2";             shift 2 ;;
      --url)                 url="$2";               shift 2 ;;
      --port)                port="$2";              shift 2 ;;
      --endpoint)            endpoint="$2";          shift 2 ;;
      --entrypoint)          entrypoint="$2";        shift 2 ;;
      --cli)                 cli="$2";               shift 2 ;;
      --setup-cmd)           setup_cmd="$2";         shift 2 ;;
      --artifact-dir-name)   artifact_dir_name="$2"; shift 2 ;;
      --)                    shift; break ;;
      --*)                   if [[ -n "${2:-}" && "${2}" != -* ]]; then shift 2; else shift 1; fi ;;
      *)                     shift ;;
    esac
  done

  log "Parsed args:
    result_dir:    $result_dir
    model:         $model
    url:           $url
    port:          $port
    endpoint:      $endpoint
    entrypoint:    $entrypoint
    setup_cmd:     ${setup_cmd:-}
    artifact_dir:  $artifact_dir_name
    cli:           ${cli:-}"
}

run_setup_cmd() {
  if [[ -z "$setup_cmd" ]]; then
    return
  fi

  log "Running accuracy setup command: $setup_cmd"
  bash -lc "$setup_cmd"
  log "Accuracy setup command complete"
}

expand_cli() {
  local artifact_dir="$1"
  local full_url="$2"
  local expanded="$cli"

  expanded="${expanded//\{model\}/$model}"
  expanded="${expanded//\{url\}/$full_url}"
  expanded="${expanded//\{endpoint\}/$endpoint}"
  expanded="${expanded//\{result_dir\}/$result_dir}"
  expanded="${expanded//\{artifact_dir\}/$artifact_dir}"
  expanded="${expanded//$'\n'/ }"

  echo "$expanded"
}

copy_accuracy_results() {
  local artifact_dir="$1"
  local accuracy_path="$artifact_dir/accuracy_results.csv"

  if [[ ! -s "$accuracy_path" ]]; then
    log "ERROR: accuracy benchmark was requested but $accuracy_path was not produced"
    exit 1
  fi

  cp "$accuracy_path" "$result_dir/accuracy_results.csv"
  log "accuracy report saved to $result_dir/accuracy_results.csv"
}

main() {
  process_args "$@"

  if [[ -z "$result_dir" ]]; then
    log "ERROR: --result-dir is required"; exit 1
  fi
  if [[ -z "$model" ]]; then
    log "ERROR: --model is required"; exit 1
  fi
  if [[ -z "$entrypoint" ]]; then
    log "ERROR: --entrypoint is required"; exit 1
  fi

  run_setup_cmd

  local full_url="${url}:${port}"
  local artifact_dir="$result_dir/$artifact_dir_name"
  rm -rf "$artifact_dir"
  mkdir -p "$artifact_dir"

  local expanded_cli
  expanded_cli="$(expand_cli "$artifact_dir" "$full_url")"

  log "Launching accuracy command: $entrypoint $expanded_cli"
  bash -lc "$entrypoint $expanded_cli"
  log "accuracy command complete"

  copy_accuracy_results "$artifact_dir"
}

main "$@"
exit 0
