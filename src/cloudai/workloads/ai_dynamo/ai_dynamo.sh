#!/usr/bin/env bash

###############################################################################
# Strict mode and globals
###############################################################################
set -Eeuo pipefail
IFS=$'\n\t'

_pids=()
RESULTS_DIR="/cloudai_run_results"
HUGGINGFACE_HOME="/root/.cache/huggingface"
DONE_MARKER="frontend_done.marker"

# Quiet logs
export DYN_SDK_DISABLE_ANSI_LOGGING=1
export VLLM_DISABLE_COLORED_OUTPUT=1
export VLLM_NO_COLOR=1
export ABSL_LOGGING_USE_COLOR=0
export DYN_LOGGING_DISABLE_ANSI_COLORS=1
export TERM=dumb
export NO_COLOR=1
export DEBIAN_FRONTEND=noninteractive
export APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# Arg buckets
declare -A prefill_args=()
declare -A decode_args=()
declare -A genai_perf_args=()
declare -A dynamo_args=(
  ["node-setup-cmd"]=""
  ["prefill-cmd"]="python3 -m dynamo.vllm --is-prefill-worker"
  ["decode-cmd"]="python3 -m dynamo.vllm"
  ["ingress-cmd"]="python -m dynamo.frontend --router-mode kv"
  ["port"]="8080"
  ["endpoint"]="v1/chat/completions"
  ["model"]="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  ["etcd-cmd"]="etcd --log-level debug"
  ["nats-cmd"]="nats-server -js"
  ["etcd-port"]="2379"
  ["nats-port"]="4222"
  ["workspace-path"]="/workspace"
  ["frontend-node"]=""
  ["num-prefill-nodes"]="1"
  ["num-decode-nodes"]="1"
  ["prefill-nodelist"]=""
  ["decode-nodelist"]=""
  ["tp-arg-name"]="tensor-parallel-size"
  ["pp-arg-name"]="pipeline-parallel-size"
  ["multiple-prefill-workers-per-node"]="true"
  ["multiple-decode-workers-per-node"]="true"
  ["prefill-initialized-regex"]="prefill.*initialized"
  ["decode-initialized-regex"]="decode.*initialized"
)

GENAI_PERF_PROFILE_EXPORT_FILE="profile.json"
GENAI_PERF_ARTIFACT_DIR="genai_perf_artifacts"

###############################################################################
# Utils
###############################################################################
log() { echo "[$(date --iso-8601=ns) $(hostname)]: $*"; }
die() { log "ERROR: $*"; exit 1; }

array_to_args() {
  local -n _arr=$1
  local out=()
  for k in "${!_arr[@]}"; do
    case "$k" in
      "--extra-args"|"--num-nodes") continue ;;
    esac
    out+=("$k" "${_arr[$k]}")
  done
  printf '%q ' "${out[@]}"
}

gpu_count() {
  local v="${CUDA_VISIBLE_DEVICES:-}"
  [[ -z "$v" ]] && { echo 0; return; }
  tr ',' '\n' <<<"$v" | wc -l
}

run_bg() {
  "$@" &
  local pid=$!
  _pids+=("$pid")
  echo "$pid"
}

cleanup() {
  if [[ ${#_pids[@]} -gt 0 ]]; then
    log "Stopping ${#_pids[@]} child processes"
    kill "${_pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

###############################################################################
# Arg parsing - split into focused steps
###############################################################################
parse_cli_pairs() {
  log "Parsing CLI pairs"
  while [[ $# -ge 2 ]]; do
    log "  $1 $2"
    case "$1" in
      --dynamo-*) dynamo_args["${1#--dynamo-}"]="$2" ;;
      --prefill-*) prefill_args["--${1#--prefill-}"]="$2" ;;
      --decode-*)  decode_args["--${1#--decode-}"]="$2" ;;
      --genai-perf-*) genai_perf_args["--${1#--genai-perf-}"]="$2" ;;
      --huggingface-home) HUGGINGFACE_HOME="$2" ;;
      --results-dir) RESULTS_DIR="$2" ;;
      *) log "Unrecognized flag preserved: $1 $2" ;;
    esac
    shift 2
  done
}

derive_node_lists() {
  if [[ -z "${dynamo_args["decode-nodelist"]}" ]]; then
    [[ -z "${DYNAMO_NODELIST:-}" ]] && die "DYNAMO_NODELIST not set and --dynamo-decode-nodelist not provided"
    dynamo_args["decode-nodelist"]="$(echo "$DYNAMO_NODELIST" | cut -d',' -f1-"${dynamo_args["num-decode-nodes"]}")"
  fi
  if [[ -z "${dynamo_args["prefill-nodelist"]}" ]]; then
    [[ -z "${DYNAMO_NODELIST:-}" ]] && die "DYNAMO_NODELIST not set and --dynamo-prefill-nodelist not provided"
    local start=$(( ${dynamo_args["num-decode-nodes"]} + 1 ))
    dynamo_args["prefill-nodelist"]="$(echo "$DYNAMO_NODELIST" | cut -d',' -f${start}-)"
  fi
  if [[ -z "${dynamo_args["frontend-node"]}" ]]; then
    dynamo_args["frontend-node"]="$(echo "${dynamo_args["decode-nodelist"]}" | cut -d',' -f1)"
  fi
}

patch_endpoint_url() {
  dynamo_args["url"]="http://${dynamo_args["frontend-node"]}:${dynamo_args["port"]}"
}

mirror_model_to_sections() {
  prefill_args["--model"]=${dynamo_args["model"]}
  decode_args["--model"]=${dynamo_args["model"]}
  genai_perf_args["--model"]=${dynamo_args["model"]}
}

patch_genai_perf_defaults() {
  genai_perf_args["--url"]=${dynamo_args["url"]}
  genai_perf_args["--endpoint"]=${dynamo_args["endpoint"]}
  genai_perf_args["--artifact-dir"]="${RESULTS_DIR}/${GENAI_PERF_ARTIFACT_DIR}/"
  genai_perf_args["--profile-export-file"]="${GENAI_PERF_PROFILE_EXPORT_FILE}"
}

derive_gpu_splits() {
  local tp="--${dynamo_args["tp-arg-name"]}"
  local pp="--${dynamo_args["pp-arg-name"]}"

  [[ -z "${prefill_args[$tp]:-}" || -z "${prefill_args[$pp]:-}" ]] && die "Prefill TP or PP not provided"
  [[ -z "${decode_args[$tp]:-}" || -z "${decode_args[$pp]:-}" ]] && die "Decode TP or PP not provided"

  dynamo_args["prefill-gpus-per-worker"]=$(( prefill_args[$tp] * prefill_args[$pp] ))
  dynamo_args["decode-gpus-per-worker"]=$(( decode_args[$tp] * decode_args[$pp] ))

  if (( dynamo_args["prefill-gpus-per-worker"] == 0 || dynamo_args["decode-gpus-per-worker"] == 0 )); then
    die "Invalid TP or PP configuration"
  fi

  local ngpu
  ngpu=$(gpu_count)
  (( ngpu > 0 )) || die "No GPUs found in CUDA_VISIBLE_DEVICES"

  if [[ "${dynamo_args["multiple-prefill-workers-per-node"]}" != "true" ]]; then
    dynamo_args["prefill-gpus-per-worker"]="$ngpu"
  fi
  if [[ "${dynamo_args["multiple-decode-workers-per-node"]}" != "true" ]]; then
    dynamo_args["decode-gpus-per-worker"]="$ngpu"
  fi

  dynamo_args["prefill-workers-per-node"]=$(( ngpu / dynamo_args["prefill-gpus-per-worker"] ))
  dynamo_args["decode-workers-per-node"]=$(( ngpu / dynamo_args["decode-gpus-per-worker"] ))
}

apply_num_node_overrides() {
  if [[ -n "${prefill_args["--num-nodes"]:-}" ]]; then
    dynamo_args["num-prefill-nodes"]=${prefill_args["--num-nodes"]}
  fi
  if [[ -n "${decode_args["--num-nodes"]:-}" ]]; then
    dynamo_args["num-decode-nodes"]=${decode_args["--num-nodes"]}
  fi
}

dump_config() {
  log "Dynamo args: $(for k in "${!dynamo_args[@]}"; do printf '%s:%s; ' "$k" "${dynamo_args[$k]}"; done)"
  log "Prefill args: $(for k in "${!prefill_args[@]}"; do printf '%s:%s; ' "$k" "${prefill_args[$k]}"; done)"
  log "Decode args: $(for k in "${!decode_args[@]}"; do printf '%s:%s; ' "$k" "${decode_args[$k]}"; done)"
  log "GenAI perf args: $(for k in "${!genai_perf_args[@]}"; do printf '%s:%s; ' "$k" "${genai_perf_args[$k]}"; done)"
}

parse_args() {
  parse_cli_pairs "$@"
  derive_node_lists
  patch_endpoint_url
  mirror_model_to_sections
  patch_genai_perf_defaults
  derive_gpu_splits
  apply_num_node_overrides
  dump_config
}

###############################################################################
# Launchers
###############################################################################
launch_node_setup_cmd() {
  local cmd="${dynamo_args["node-setup-cmd"]}"
  if [[ -n "$cmd" ]]; then
    log "Node setup: $cmd"
    bash -c "$cmd"
    log "Node setup complete"
  fi
}

launch_etcd() {
  log "Launching etcd on port ${dynamo_args["etcd-port"]}"
  run_bg \
    ${dynamo_args["etcd-cmd"]} \
      --listen-client-urls "http://0.0.0.0:${dynamo_args["etcd-port"]}" \
      --advertise-client-urls "http://0.0.0.0:${dynamo_args["etcd-port"]}" \
      > "${RESULTS_DIR}/etcd.log" 2>&1 >/dev/null
}

launch_nats() {
  log "Launching nats on port ${dynamo_args["nats-port"]}"
  run_bg \
    ${dynamo_args["nats-cmd"]} -p "${dynamo_args["nats-port"]}" \
    > "${RESULTS_DIR}/nats.log" 2>&1 >/dev/null
}

wait_for_etcd() {
  local url="${ETCD_ENDPOINTS}/readyz"
  until [[ "$(curl -ks "$url" || true)" == "ok" ]]; do
    log "Waiting for etcd readiness $url"
    sleep 10
  done
  log "etcd is ready"
}

launch_ingress() {
  log "Launching ingress: ${dynamo_args["ingress-cmd"]} --http-port ${dynamo_args["port"]}"
  run_bg \
    ${dynamo_args["ingress-cmd"]} --http-port "${dynamo_args["port"]}" \
    > "${RESULTS_DIR}/dynamo_ingress.log" 2>&1 >/dev/null
}

launch_workers() {
  local role="$1"  # prefill or decode
  local cmd_key="${role}-cmd"
  local gpus_per_worker_key="${role}-gpus-per-worker"
  local workers_per_node_key="${role}-workers-per-node"
  local log_prefix="dynamo_${role}_${SLURM_NODEID:-0}_"

  wait_for_etcd

  local workers=${dynamo_args[$workers_per_node_key]}
  local gpus_per_worker=${dynamo_args[$gpus_per_worker_key]}
  local cmd="${dynamo_args[$cmd_key]}"

  (( workers > 0 )) || die "Computed 0 workers for $role"
  (( gpus_per_worker > 0 )) || die "Computed 0 GPUs per worker for $role"

  for (( i=0; i<workers; i++ )); do
    local start=$(( 1 + i * gpus_per_worker ))
    local end=$(( start + gpus_per_worker - 1 ))
    local gpu_list
    gpu_list="$(echo "${CUDA_VISIBLE_DEVICES:-}" | cut -d',' -f${start}-${end})"
    local log_file="${RESULTS_DIR}/${log_prefix}${i}.log"

    log "Launching ${role} worker ${i} on GPUs ${gpu_list}"

    if [[ "$role" == "prefill" ]]; then
      local args
      args=$(array_to_args prefill_args)
      CUDA_VISIBLE_DEVICES="$gpu_list" run_bg bash -c \
        "${cmd} ${args} ${prefill_args["--extra-args"]:-}" \
        > "$log_file" 2>&1
    else
      local args
      args=$(array_to_args decode_args)
      CUDA_VISIBLE_DEVICES="$gpu_list" run_bg bash -c \
        "${cmd} ${args} ${decode_args["--extra-args"]:-}" \
        > "$log_file" 2>&1
    fi
  done
}

exit_on_error() {
  local zmq_cnt ucx_cnt
  zmq_cnt=$(grep -il "zmq.error.ZMQError: Address already in use" "${RESULTS_DIR}"/dynamo_*.log 2>/dev/null | wc -l | tr -d ' ')
  if (( zmq_cnt > 0 )); then
    die "ZMQ ERROR: Found ${zmq_cnt} failed workers"
  fi
  ucx_cnt=$(grep -il "UCX.*ERROR" "${RESULTS_DIR}"/ucx_log_*.log 2>/dev/null | wc -l | tr -d ' ')
  if (( ucx_cnt > 0 )); then
    die "UCX ERROR: Found ${ucx_cnt} failed workers"
  fi
}

wait_for_dynamo_frontend() {
  local total_prefill=$(( ${dynamo_args["num-prefill-nodes"]} * ${dynamo_args["prefill-workers-per-node"]} ))
  local total_decode=$(( ${dynamo_args["num-decode-nodes"]} * ${dynamo_args["decode-workers-per-node"]} ))

  while true; do
    local in_p in_d
    in_p=$(grep -il "${dynamo_args["prefill-initialized-regex"]}" "${RESULTS_DIR}"/'*prefill*' 2>/dev/null | wc -l | tr -d ' ')
    in_d=$(grep -il "${dynamo_args["decode-initialized-regex"]}" "${RESULTS_DIR}"/'*decode*' 2>/dev/null | wc -l | tr -d ' ')
    if (( in_p == total_prefill && in_d == total_decode )); then
      break
    fi
    log "Initialized ${in_p}/${total_prefill} prefill. ${in_d}/${total_decode} decode."
    exit_on_error
    sleep 30
  done
  log "Dynamo frontend is ready"
}

wait_for_frontend_marker() {
  while [[ ! -f "$DONE_MARKER" ]]; do
    exit_on_error
    log "Waiting for frontend done marker: $DONE_MARKER"
    sleep 30
  done
  log "Done marker found"
}

launch_genai_perf() {
  wait_for_dynamo_frontend

  local payload
  payload=$(cat <<EOF
{
  "model": "${dynamo_args["model"]}",
  "messages": [{"role": "user", "content": "The color of sky is"}],
  "stream": false,
  "max_tokens": 10
}
EOF
)
  local url="${dynamo_args["url"]}/v1/chat/completions"
  local resp
  resp=$(curl -s -X POST "$url" -H "Content-Type: application/json" -d "$payload" || true)
  echo "Response: $resp"

  local args
  args=$(array_to_args genai_perf_args)
  log "Launching genai-perf: ${dynamo_args["genai-perf-cmd"]} ${args} ${genai_perf_args["--extra-args"]:-}"
  bash -c "${dynamo_args["genai-perf-cmd"]} ${args} ${genai_perf_args["--extra-args"]:-}" \
    > "${RESULTS_DIR}/genai_perf.log" 2>&1
  log "genai-perf completed"
}

###############################################################################
# Main
###############################################################################
main() {
  export HF_HOME="${HUGGINGFACE_HOME}"
  export NATS_SERVER="nats://${dynamo_args["frontend-node"]}:${dynamo_args["nats-port"]}"
  export ETCD_ENDPOINTS="http://${dynamo_args["frontend-node"]}:${dynamo_args["etcd-port"]}"
  export UCX_LOG_FILE="${RESULTS_DIR}/ucx_log_%h.log"

  DONE_MARKER="${RESULTS_DIR}/${DONE_MARKER}"
  mkdir -p "${RESULTS_DIR}"

  launch_node_setup_cmd
  cd "${dynamo_args["workspace-path"]}"

  local nodename="${SLURMD_NODENAME:-$(hostname)}"
  local nodeid="${SLURM_NODEID:-0}"
  log "Node ID: ${nodeid}. Host: ${nodename}"

  if [[ "${dynamo_args["frontend-node"]}" == *"$nodename"* ]]; then
    log "Role: frontend"
    launch_etcd
    launch_nats
    wait_for_etcd
    launch_ingress
  fi

  if [[ "${dynamo_args["decode-nodelist"]}" == *"$nodename"* ]]; then
    log "Role: decode"
    launch_workers "decode"
  fi

  if [[ "${dynamo_args["prefill-nodelist"]}" == *"$nodename"* ]]; then
    log "Role: prefill"
    launch_workers "prefill"
  fi

  if [[ "${dynamo_args["frontend-node"]}" == *"$nodename"* ]]; then
    launch_genai_perf
    : > "$DONE_MARKER"
  fi

  wait_for_frontend_marker
}

###############################################################################
# Entry
###############################################################################
parse_args "$@"
log "env: $(env)"
log "Starting main"
main
log "Done with main"
