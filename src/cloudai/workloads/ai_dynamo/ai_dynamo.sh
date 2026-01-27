#!/bin/bash

# CloudAI params
RESULTS_DIR="/cloudai_run_results"
HUGGINGFACE_HOME="/root/.cache/huggingface"
DONE_MARKER="dynamo_frontend_done.marker"
FATAL_ERROR_MARKER="dynamo_fatal_error.marker"
NODE_ROLES_FILE="node_roles.log"
genai_perf_wrapper_script="/cloudai_install/genai_perf_wrapper.sh"
calc_percentile_csv_script="/cloudai_install/calc_percentile_csv.py"
genai_perf_report_file="genai_perf_report.csv"

export DYN_SDK_DISABLE_ANSI_LOGGING=1
export VLLM_DISABLE_COLORED_OUTPUT=1
export VLLM_NO_COLOR=1
export ABSL_LOGGING_USE_COLOR=0
export DYN_LOGGING_DISABLE_ANSI_COLORS=1

export TERM=dumb
export NO_COLOR=1

export DEBIAN_FRONTEND=noninteractive
export APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

declare -A prefill_args
declare -A decode_args
declare -A lmcache_args
declare -A lmcache_config
declare -A genai_perf_args
declare -A genai_perf_config
declare -A lmbench_args
declare -A lmbench_config
declare -A custom_bench_args
declare -A custom_bench_config

declare -A dynamo_args
dynamo_args["backend"]="vllm"
dynamo_args["node-setup-cmd"]=""
dynamo_args["prefill-cmd"]="python3 -m dynamo.vllm --is-prefill-worker"
dynamo_args["decode-cmd"]="python3 -m dynamo.vllm"
dynamo_args["ingress-cmd"]="python -m dynamo.frontend --router-mode kv"
dynamo_args["port"]=$((8080 + SLURM_JOBID % 100))
dynamo_args["endpoint"]="v1/chat/completions"
dynamo_args["model"]="Qwen/Qwen3-0.6B"
dynamo_args["connector"]="none"
dynamo_args["etcd-port"]=2379
dynamo_args["nats-port"]=4222
dynamo_args["workspace-path"]="/workspace"
dynamo_args["frontend-node"]=""
dynamo_args["num-prefill-nodes"]=1
dynamo_args["num-decode-nodes"]=1
dynamo_args["prefill-nodes"]=""
dynamo_args["decode-nodes"]=""
dynamo_args["tp-arg-name"]="tensor-parallel-size"
dynamo_args["pp-arg-name"]="pipeline-parallel-size"
dynamo_args["multiple-prefill-workers-per-node"]="true"
dynamo_args["multiple-decode-workers-per-node"]="true"
dynamo_args["prefill-initialized-regex"]="Worker.*has.been.initialized"
dynamo_args["decode-initialized-regex"]="Worker.*has.been.initialized"

dynamo_args["etcd-cmd"]="etcd --log-level debug"
dynamo_args["nats-cmd"]="nats-server -js"
dynamo_args["genai-perf-cmd"]="genai-perf profile"
dynamo_args["worker-error-pattern"]="zmq.error.ZMQError:.Address.already.in.use|UCX.*ERROR|ERROR.core.run_engine_core:.EngineCore.failed.to.start|ERROR.multiproc_executor.worker_busy_loop:.WorkerProc.hit.an.exception|EngineDeadError|EngineCore.encountered.an.issue"

# sglang-specific optional ports. Ignored by vllm.
dynamo_args["sgl-http-port"]=9001
dynamo_args["prefill-port"]=30011
dynamo_args["decode-port"]=30021


function log()
{
  echo "[$(date +%F\ %T) $(hostname)]: " "$@"
}

function min()
{
  echo "$(( $1 < $2 ? $1 : $2 ))"
}

_is_vllm() { [[ "${dynamo_args["backend"]}" == "vllm" ]]; }
_is_sglang() { [[ "${dynamo_args["backend"]}" == "sglang" ]]; }

_csv_len() { grep -oE '[^,]+' <<< "$1" | wc -l; }

_first_in_csv() { echo "$1" | cut -d',' -f1; }

_csv_index_of() {
  local list="$1" name="$2"
  local IFS=',' arr i
  read -ra arr <<< "$list"
  for i in "${!arr[@]}"; do
    if [[ "${arr[$i]}" == "$name" || "${arr[$i]}" == *"$name"* || "$name" == *"${arr[$i]}"* ]]; then
      echo "$i"; return 0
    fi
  done
  echo "-1"
}

_validate_or_build_nodelists() {
  local dl_len=$(_csv_len "${dynamo_args["decode-nodes"]}")
  local pl_len=$(_csv_len "${dynamo_args["prefill-nodes"]}")
  if (( dl_len > 0 )); then dynamo_args["num-decode-nodes"]="$dl_len"; fi
  if (( pl_len > 0 )); then dynamo_args["num-prefill-nodes"]="$pl_len"; fi

  if [[ -z "${dynamo_args["decode-nodes"]}" || -z "${dynamo_args["prefill-nodes"]}" ]]; then
    if [[ -z "${DYNAMO_NODELIST:-}" ]]; then
      log "ERROR: Provide --dynamo-decode-nodes/--dynamo-prefill-nodes or set DYNAMO_NODELIST"; exit 1
    fi
    local d="${dynamo_args["num-decode-nodes"]}"
    local p="${dynamo_args["num-prefill-nodes"]}"
    local total=$(_csv_len "${DYNAMO_NODELIST}")
    if (( total < d + p )); then
      log "ERROR: DYNAMO_NODELIST has ${total} entries; need decode(${d})+prefill(${p})"; exit 1
    fi
    [[ -z "${dynamo_args["decode-nodes"]}" ]] && \
      dynamo_args["decode-nodes"]=$(echo "$DYNAMO_NODELIST" | cut -d',' -f1-"$d")
    [[ -z "${dynamo_args["prefill-nodes"]}" ]] && \
      dynamo_args["prefill-nodes"]=$(echo "$DYNAMO_NODELIST" | cut -d',' -f$(( d + 1 ))-)
  fi
}

_gpus_per_node() {
  local n=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c . || true)
  [[ "$n" -gt 0 ]] && echo "$n" || echo "1"
}

_resolve_host_ip() {
  local host="$1"
  local ip
  ip="$(getent ahosts "$host" | grep STREAM | head -n1 | awk '{print $1}')"
  if [[ -z "$ip" ]]; then
    log "ERROR: Could not resolve IP for host $host"
    exit 1
  fi
  echo "$ip"
}

_apply_sglang_section_args() {
  local self="$(_current_node_name)"
  local gpn="$(_gpus_per_node)"

  # prefill group
  local prefill_nodes="${dynamo_args["num-prefill-nodes"]}"
  local prefill_master_host="$(_first_in_csv "${dynamo_args["prefill-nodes"]}")"
  local prefill_master_ip="$(_resolve_host_ip "${prefill_master_host}")"
  local prefill_rank="$(_csv_index_of "${dynamo_args["prefill-nodes"]}" "$self")"
  local prefill_total_gpus=$(( gpn * prefill_nodes ))
  prefill_args["--dist-init-addr"]="${prefill_master_ip}:${dynamo_args["prefill-port"]}"
  prefill_args["--nnodes"]="${prefill_nodes}"
  prefill_args["--node-rank"]="$([[ "$prefill_rank" -ge 0 ]] && echo "$prefill_rank" || echo 0)"
  prefill_args["--tp-size"]="${prefill_args["--tp-size"]:-${prefill_total_gpus}}"
  prefill_args["--dp-size"]="${prefill_args["--dp-size"]:-${prefill_total_gpus}}"

  # decode group
  local decode_nodes="${dynamo_args["num-decode-nodes"]}"
  local decode_master_host="$(_first_in_csv "${dynamo_args["decode-nodes"]}")"
  local decode_master_ip="$(_resolve_host_ip "${decode_master_host}")"
  local decode_rank="$(_csv_index_of "${dynamo_args["decode-nodes"]}" "$self")"
  local decode_total_gpus=$(( gpn * decode_nodes ))
  decode_args["--dist-init-addr"]="${decode_master_ip}:${dynamo_args["decode-port"]}"
  decode_args["--nnodes"]="${decode_nodes}"
  decode_args["--node-rank"]="$([[ "$decode_rank" -ge 0 ]] && echo "$decode_rank" || echo 0)"
  decode_args["--tp-size"]="${decode_args["--tp-size"]:-${decode_total_gpus}}"
  decode_args["--dp-size"]="${decode_args["--dp-size"]:-${decode_total_gpus}}"

  if [[ -n "${dynamo_args["deepep-config"]:-}" ]]; then
    [[ -f "${dynamo_args["deepep-config"]}" ]] || log "WARN: deepep-config not found: ${dynamo_args["deepep-config"]}"
    prefill_args["--deepep-config"]="${dynamo_args["deepep-config"]}"
    decode_args["--deepep-config"]="${dynamo_args["deepep-config"]}"
  fi

  unset 'prefill_args["--model"]'
  unset 'decode_args["--model"]'
}

_apply_genai_perf_section_args() {
  if [[ ! -v genai_perf_args["--warmup-request-count"] ]]; then
    genai_perf_args["--warmup-request-count"]=$(( ${dynamo_args["warmup-request-multiplier"]} * ${genai_perf_args["--concurrency"]} ))
    genai_perf_args["--warmup-request-count"]=$(min ${genai_perf_args["--warmup-request-count"]} ${dynamo_args["min-warmup-request-count"]})
  fi

  if [[ ! -v genai_perf_args["--request-count"] ]]; then
    genai_perf_args["--request-count"]=$(( ${dynamo_args["actual-request-multiplier"]} * ${genai_perf_args["--concurrency"]} ))
    genai_perf_args["--request-count"]=$(min ${genai_perf_args["--request-count"]} ${dynamo_args["min-request-count"]})
  fi
}

_parse_cli_pairs() {
  log "Parsing args:"
  while [[ $# -ge 2 ]]; do
    echo "  $1 $2"
    key="$1"
    case $key in
      --dynamo-*)
        dynamo_args["${key#--dynamo-}"]="$2" ;;
      --prefill-*)
        prefill_args["--${key#--prefill-}"]="$2" ;;
      --decode-*)
        decode_args["--${key#--decode-}"]="$2" ;;
      --lmcache-args-*)
        lmcache_args["${key#--lmcache-args-}"]="$2" ;;
      --lmcache-*)
        lmcache_config["${key#--lmcache-}"]="$2" ;;
      --lmbench-args-*)
        lmbench_args["${key#--lmbench-args-}"]="$2" ;;
      --lmbench-*)
        lmbench_config["${key#--lmbench-}"]="$2" ;;
      --genai_perf-args-*)
        genai_perf_args["${key#--genai-perf-args-}"]="$2" ;;
      --genai-perf-*)
        genai_perf_config["${key#--genai-perf-}"]="$2" ;;
      --custom-bench-args-*)
        custom_bench_args["${key#--custom-bench-args-}"]="$2" ;;
      --custom-bench-*)
        custom_bench_config["${key#--custom-bench-}"]="$2" ;;
      --huggingface-home)
        HUGGINGFACE_HOME="$2" ;;
      --results-dir)
        RESULTS_DIR="$2" ;;
      --genai_perf_wrapper_script)
        genai_perf_wrapper_script="$2" ;;
      --calc_percentile_csv_script)
        calc_percentile_csv_script="$2" ;;
      --genai_perf_report_file)
        genai_perf_report_file="$2" ;;
    esac
    shift; shift;
  done
}

_set_backend_defaults() {
  case "${dynamo_args["backend"]}" in
    vllm)
      :
      ;;
    sglang)
      dynamo_args["prefill-cmd"]="python3 -m dynamo.sglang.worker"
      dynamo_args["decode-cmd"]="python3 -m dynamo.sglang.decode_worker"
      dynamo_args["ingress-cmd"]="python3 -m dynamo.frontend"
      ;;
    *)
      log "ERROR: Unknown backend '${dynamo_args["backend"]}'"
      exit 1
      ;;
  esac
}

_apply_connector_settings() {
  local connector="${dynamo_args["connector"]:-}"
  if [[ -z "$connector" || "$connector" == "none" ]]; then
    ENABLE_LMCACHE="${ENABLE_LMCACHE:-0}"
    ENABLE_KVBM="${ENABLE_KVBM:-0}"
    return
  fi

  case "$connector" in
    lmcache)
      ENABLE_LMCACHE=1
      ENABLE_KVBM=0
      ;;
    kvbm)
      ENABLE_LMCACHE=0
      ENABLE_KVBM=1
      ;;
    *)
      log "ERROR: Unknown connector '${connector}' (expected none|lmcache|kvbm)"
      exit 1
      ;;
  esac
}

_sync_num_nodes_from_section_args() {
  if [[ -n "${prefill_args["--num-nodes"]:-}" ]]; then
    dynamo_args["num-prefill-nodes"]="${prefill_args["--num-nodes"]}"
  fi
  if [[ -n "${decode_args["--num-nodes"]:-}" ]]; then
    dynamo_args["num-decode-nodes"]="${decode_args["--num-nodes"]}"
  fi
}

_patch_dynamo_args() {
  if [[ -z "${dynamo_args["decode-nodes"]}" ]]; then
    if [[ -n "${decode_args["--node-list"]}" ]]; then
      dynamo_args["decode-nodes"]="${decode_args["--node-list"]}"
    else
      dynamo_args["decode-nodes"]=$(echo $DYNAMO_NODELIST | cut -d',' -f1-${dynamo_args["num-decode-nodes"]})
    fi
  fi

  if [[ -z "${dynamo_args["prefill-nodes"]}" ]]; then
    if [[ -n "${prefill_args["--node-list"]}" ]]; then
      dynamo_args["prefill-nodes"]="${prefill_args["--node-list"]}"
    else
      dynamo_args["prefill-nodes"]=$(echo $DYNAMO_NODELIST | cut -d',' -f$(( ${dynamo_args["num-decode-nodes"]} + 1 ))-)
    fi
  fi

  if [[ -z "${dynamo_args["frontend-node"]}" ]]; then
    dynamo_args["frontend-node"]=$(echo ${dynamo_args["decode-nodes"]} | cut -d',' -f1)
  fi

  dynamo_args["url"]="http://${dynamo_args["frontend-node"]}:${dynamo_args["port"]}"

  _validate_or_build_nodelists
}

_patch_section_args() {
  if _is_sglang; then
    _apply_sglang_section_args
  fi

  _apply_genai_perf_section_args
}

_compute_worker_allocation_sglang() {
  local num_gpus="$(_gpus_per_node)"
  if [[ $num_gpus -eq 0 ]]; then
    log "ERROR: No GPUs found in CUDA_VISIBLE_DEVICES"
    exit 1
  fi

  # sglang: one worker per node using all GPUs
  dynamo_args["prefill-gpus-per-worker"]=$num_gpus
  dynamo_args["decode-gpus-per-worker"]=$num_gpus
  dynamo_args["prefill-workers-per-node"]=1
  dynamo_args["decode-workers-per-node"]=1

  if [[ -n "${prefill_args["--num-nodes"]}" ]]; then
    dynamo_args["num-prefill-nodes"]=${prefill_args["--num-nodes"]}
  fi
  if [[ -n "${decode_args["--num-nodes"]}" ]]; then
    dynamo_args["num-decode-nodes"]=${decode_args["--num-nodes"]}
  fi
}

_compute_worker_allocation_vllm() {
  local tp_arg_name="--${dynamo_args["tp-arg-name"]}"
  local pp_arg_name="--${dynamo_args["pp-arg-name"]}"
  local num_gpus="$(_gpus_per_node)"

  if [[ $num_gpus -eq 0 ]]; then
    log "ERROR: No GPUs found in CUDA_VISIBLE_DEVICES"
    exit 1
  fi

  dynamo_args["prefill-gpus-per-worker"]=$(( prefill_args[$tp_arg_name] * prefill_args[$pp_arg_name] ))
  dynamo_args["decode-gpus-per-worker"]=$(( decode_args[$tp_arg_name] * decode_args[$pp_arg_name] ))

  if [[ ${dynamo_args["prefill-gpus-per-worker"]} -eq 0 ]] || [[ ${dynamo_args["decode-gpus-per-worker"]} -eq 0 ]]; then
    log "ERROR: Invalid TP/PP configuration"
    exit 1
  fi

  if [[ "${dynamo_args["multiple-prefill-workers-per-node"]}" != "true" ]]; then
    dynamo_args["prefill-gpus-per-worker"]=$num_gpus
  fi

  if [[ "${dynamo_args["multiple-decode-workers-per-node"]}" != "true" ]]; then
    dynamo_args["decode-gpus-per-worker"]=$num_gpus
  fi

  log "DECODE: num GPUs: $num_gpus, GPUs per worker: ${dynamo_args["decode-gpus-per-worker"]}"
  log "PREFILL: num GPUs: $num_gpus, GPUs per worker: ${dynamo_args["prefill-gpus-per-worker"]}"
  dynamo_args["prefill-workers-per-node"]=$(( num_gpus / dynamo_args["prefill-gpus-per-worker"] ))
  dynamo_args["decode-workers-per-node"]=$(( num_gpus / dynamo_args["decode-gpus-per-worker"] ))
  log "DECODE: workers per node: ${dynamo_args["decode-workers-per-node"]}"
  log "PREFILL: workers per node: ${dynamo_args["prefill-workers-per-node"]}"

  if [[ -n "${prefill_args["--num-nodes"]}" ]]; then
    dynamo_args["num-prefill-nodes"]=${prefill_args["--num-nodes"]}
  fi
  if [[ -n "${decode_args["--num-nodes"]}" ]]; then
    dynamo_args["num-decode-nodes"]=${decode_args["--num-nodes"]}
  fi
  log "NUM PREFILL NODES: ${dynamo_args["num-prefill-nodes"]}"
  log "NUM DECODE NODES: ${dynamo_args["num-decode-nodes"]}"
}

_compute_worker_allocation() {
  if _is_sglang; then
    _compute_worker_allocation_sglang
  else
    _compute_worker_allocation_vllm
  fi
}

_dump_args() {
  log "Dynamo args: $(for key in "${!dynamo_args[@]}"; do echo -n "$key: ${dynamo_args[$key]}; "; done)"
  log "Prefill args: $(for key in "${!prefill_args[@]}"; do echo -n "$key: ${prefill_args[$key]}; "; done)"
  log "Decode args: $(for key in "${!decode_args[@]}"; do echo -n "$key: ${decode_args[$key]}; " ; done)"
  log "GenAI perf args: $(for key in "${!genai_perf_args[@]}"; do echo -n "$key: ${genai_perf_args[$key]}; "; done)"
  log "LMBench args: $(for key in "${!lmbench_args[@]}"; do echo -n "$key: ${lmbench_args[$key]}; "; done)"
}

function parse_args()
{
  _parse_cli_pairs "$@"
  _set_backend_defaults
  _sync_num_nodes_from_section_args
  _patch_dynamo_args
  _patch_section_args
  _apply_connector_settings
  _compute_worker_allocation
  _dump_args
}

function replace_placeholders() {
  local val="$1"
  val=${val//%MODEL%/${dynamo_args["model"]}}
  val=${val//%PORT%/${dynamo_args["port"]}}
  val=${val//%URL%/${dynamo_args["url"]}}
  val=${val//%ENDPOINT%/${dynamo_args["endpoint"]}}
  val=${val//%RESULTS_DIR%/${RESULTS_DIR}}
  val=${val//%HUGGINGFACE_HOME%/${HUGGINGFACE_HOME}}
  echo "$val"
}

function array_to_args()
{
  local -n arr=$1
  local result=""
  for key in "${!arr[@]}"; do
    if [[ "$key" == "--extra-args" ]] || \
       [[ "$key" == "--num-nodes" ]] || \
       [[ "$key" == "--nodes" ]]; then
      continue
    fi

    shopt -s nocasematch
    val=$(replace_placeholders "${arr[$key]}")
    result+="${key} ${val} "
  done
  echo "$result"
}

_detect_fatal_once() {
  # Only treat as fatal on vllm
  _is_vllm || return 0
  local n=0
  # Worker logs and UCX logs
  n=$(( n + $(grep -E "${dynamo_args["worker-error-pattern"]}" "${RESULTS_DIR}"/dynamo_*.log 2>/dev/null | wc -l || true) ))
  n=$(( n + $(grep -E "UCX.*ERROR" "${RESULTS_DIR}"/ucx_log_*.log 2>/dev/null | wc -l || true) ))
  echo "${n}"
}

function perform_exit()
{
  local exit_code=$1
  local sleep_before_exit="${dynamo_args["sleep-before-exit"]}"
  if [[ -n "${sleep_before_exit}" ]]; then
    log "Sleeping for ${sleep_before_exit} seconds before exit"
    sleep "${sleep_before_exit}"
  fi
  exit "${exit_code}"
}

exit_on_error() {
  local fatal=$(_detect_fatal_once)
  if [ -f "${DONE_MARKER}" ]; then
    log "DONE_MARKER found. Skipping error check."
    return
  fi
  if [[ "${fatal}" -gt 0 ]]; then
    log "FATAL: detected ${fatal} fatal error line(s). Writing ${FATAL_ERROR_MARKER} and terminating."
    sleep 1

    touch "${FATAL_ERROR_MARKER}"
    grep -E "${dynamo_args["worker-error-pattern"]}|UCX.*ERROR" "${RESULTS_DIR}"/*.log 2>/dev/null > "${FATAL_ERROR_MARKER}"
    # Try to stop background jobs for a cleaner exit, but do not loop
    kill $(jobs -p) 2>/dev/null || true
    # Exit non-zero so srun can retry
    perform_exit 1
  fi
}

_total_workers_prefill() {
  echo $(( dynamo_args["num-prefill-nodes"] * dynamo_args["prefill-workers-per-node"] ))
}

_total_workers_decode() {
  echo $(( dynamo_args["num-decode-nodes"] * dynamo_args["decode-workers-per-node"] ))
}

_count_initialized_prefill() {
  grep -i -l -E "${dynamo_args["prefill-initialized-regex"]}" "${RESULTS_DIR}"/dynamo_*prefill* 2>/dev/null | wc -l
}

_count_initialized_decode() {
  grep -i -l -E "${dynamo_args["decode-initialized-regex"]}" "${RESULTS_DIR}"/dynamo_*decode* 2>/dev/null | wc -l
}

_expected_ready_prefill() {
  if _is_sglang; then
    echo 1
  else
    echo "$(_total_workers_prefill)"
  fi
}

_expected_ready_decode() {
  if _is_sglang; then
    echo 1
  else
    echo "$(_total_workers_decode)"
  fi
}
_gpu_list_for_worker() {
  local per_worker=$1
  local idx=$2
  local start=$(( 1 + (idx * per_worker) ))
  local end=$(( start + per_worker - 1 ))
  echo "$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f${start}-${end})"
}

_log_file_for_worker() {
  local role="$1"
  local idx="$2"
  echo "${RESULTS_DIR}/dynamo_${role}_${SLURM_NODEID}_${idx}.log"
}

function log_node_role()
{
  local node_name=$1
  local role=$2
  local roles_file="${RESULTS_DIR}/${NODE_ROLES_FILE}"
  echo "${node_name},${role}" >> "$roles_file"
}

_current_node_name() {
  echo "${SLURMD_NODENAME:-$(hostname)}"
}

_is_frontend_node() {
  local name="$(_current_node_name)"
  [[ "${dynamo_args["frontend-node"]}" == *"$name"* ]]
}

_is_decode_node() {
  local name="$(_current_node_name)"
  [[ "${dynamo_args["decode-nodes"]}" == *"$name"* ]]
}

_is_prefill_node() {
  local name="$(_current_node_name)"
  [[ "${dynamo_args["prefill-nodes"]}" == *"$name"* ]]
}

_is_genai_perf_workload() {
  [[ "${dynamo_args["workload-type"]}" == "genai-perf" ]]
}

_is_lmbench_workload() {
  [[ "${dynamo_args["workload-type"]}" == "lmbench" ]]
}

_is_single_shot_workload() {
  [[ "${dynamo_args["workload-type"]}" == "single-shot" ]]
}

_init_runtime_env() {
  if _is_vllm; then
    export HF_HOME="${HUGGINGFACE_HOME}"
  fi
  export NATS_SERVER="nats://${dynamo_args["frontend-node"]}:${dynamo_args["nats-port"]}"
  export ETCD_ENDPOINTS="http://${dynamo_args["frontend-node"]}:${dynamo_args["etcd-port"]}"
  export UCX_LOG_FILE="${RESULTS_DIR}/ucx_log_%h.log"
  DONE_MARKER="${RESULTS_DIR}/${DONE_MARKER}"
  FATAL_ERROR_MARKER="${RESULTS_DIR}/${FATAL_ERROR_MARKER}"
  rm -f "${FATAL_ERROR_MARKER}" 2>/dev/null || true
}

function launch_node_setup_cmd()
{
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.local/bin/env

  if [[ -n "${dynamo_args["node-setup-cmd"]}" ]]; then
    log "Launching node setup command: ${dynamo_args["node-setup-cmd"]}"
    bash -c "${dynamo_args["node-setup-cmd"]}"
    log "Node setup complete"
  fi
}

_require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    log "ERROR: Required command '$cmd' not found in PATH"
    exit 1
  fi
}

_ensure_dir_writable() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    log "Creating directory: $dir"
    mkdir -p "$dir" || { log "ERROR: Failed to create $dir"; exit 1; }
  fi
  if [[ ! -w "$dir" ]]; then
    log "ERROR: Directory not writable: $dir"
    exit 1
  fi
}

_port_in_use() {
  local port="$1"
  # Prefer ss. Fallback to netstat or lsof. Final fallback is nc.
  if command -v ss >/dev/null 2>&1; then
    ss -lnt "( sport = :$port )" | awk 'NR>1{exit 0} END{exit 1}'
    return $?
  elif command -v netstat >/dev/null 2>&1; then
    netstat -lnt 2>/dev/null | awk -v p=":$port" '$4 ~ p {found=1} END{exit !found}'
    return $?
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  elif command -v nc >/dev/null 2>&1; then
    nc -z 127.0.0.1 "$port" >/dev/null 2>&1
    return $?
  fi
  # If we cannot check, assume free
  return 1
}

_check_free_port_or_die() {
  local name="$1" port="$2"
  if _port_in_use "$port"; then
    log "ERROR: Port $port for $name is already in use on $(hostname)"
    exit 1
  fi
}

validate_environment() {
  log "Validating environment..."

  # Core commands needed by this script
  _require_cmd bash
  _require_cmd awk
  _require_cmd grep
  _require_cmd cut
  _require_cmd curl

  # Runtime commands invoked later
  _require_cmd python3
  _require_cmd ${dynamo_args["etcd-cmd"]%% *}     # first token if args included
  _require_cmd ${dynamo_args["nats-cmd"]%% *}

  # Basic env presence
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    log "ERROR: CUDA_VISIBLE_DEVICES is not set"
    exit 1
  fi

  # If both nodelists are empty, DYNAMO_NODELIST must be provided
  if [[ -z "${dynamo_args["decode-nodes"]}" && -z "${dynamo_args["prefill-nodes"]}" ]]; then
    if [[ -z "${DYNAMO_NODELIST:-}" ]]; then
      log "ERROR: When neither --dynamo-decode-nodes nor --dynamo-prefill-nodes is provided, DYNAMO_NODELIST must be set"
      exit 1
    fi
  fi

  # Directories
  _ensure_dir_writable "$RESULTS_DIR"
  if _is_vllm; then
    _ensure_dir_writable "$HUGGINGFACE_HOME"
  fi

  # Disk space check for RESULTS_DIR. Require at least ~1 GB free.
  local avail_kb
  avail_kb=$(df -Pk "$RESULTS_DIR" | awk 'NR==2{print $4}')
  if [[ -n "$avail_kb" && "$avail_kb" -lt 1048576 ]]; then
    log "ERROR: Less than 1 GB free in $RESULTS_DIR"
    exit 1
  fi

  # SLURM hints
  if [[ -z "${SLURM_NODEID:-}" ]]; then
    log "WARN: SLURM_NODEID is not set"
  fi
  if [[ -z "${SLURMD_NODENAME:-}" ]]; then
    log "WARN: SLURMD_NODENAME is not set. Falling back to hostname where applicable"
  fi

  # Frontend node only checks
  if _is_frontend_node; then
    # Ports must be free before we launch services
    _check_free_port_or_die "etcd"  "${dynamo_args["etcd-port"]}"
    _check_free_port_or_die "nats"  "${dynamo_args["nats-port"]}"
    _check_free_port_or_die "ingress http" "${dynamo_args["port"]}"
  fi

  # GPU count sanity
  local num_gpus="$(_gpus_per_node)"
  if [[ "$num_gpus" -le 0 ]]; then
    log "ERROR: Parsed zero GPUs from CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'"
    exit 1
  fi

  log "Environment validation complete"
}

function launch_etcd()
{
  log "Launching etcd with cmd: ${dynamo_args["etcd-cmd"]} --listen-client-urls http://0.0.0.0:${dynamo_args["etcd-port"]} --advertise-client-urls http://0.0.0.0:${dynamo_args["etcd-port"]}"
  ${dynamo_args["etcd-cmd"]} \
    --listen-client-urls http://0.0.0.0:${dynamo_args["etcd-port"]} \
    --advertise-client-urls http://0.0.0.0:${dynamo_args["etcd-port"]} \
    > ${RESULTS_DIR}/etcd.log 2>&1
}

function launch_nats()
{
  log "Launching nats with cmd: ${dynamo_args["nats-cmd"]} -p ${dynamo_args["nats-port"]}"
  ${dynamo_args["nats-cmd"]} -p ${dynamo_args["nats-port"]} > ${RESULTS_DIR}/nats.log 2>&1
}

function launch_ingress()
{
  log "Launching ingress with cmd: ${dynamo_args["ingress-cmd"]} --http-port ${dynamo_args["port"]}"
  ${dynamo_args["ingress-cmd"]} --http-port ${dynamo_args["port"]} > ${RESULTS_DIR}/dynamo_ingress.log 2>&1
}

launch_sgl_http_server() {
  local script_path="${dynamo_args["sgl-http-server-script"]}"
  local port="${dynamo_args["sgl-http-port"]}"
  if [[ -n "${script_path}" && -f "${script_path}" ]]; then
    log "Starting SGL HTTP server: ${script_path} --ns dynamo --port ${port}"
    nohup python3 "${script_path}" --ns dynamo --port "${port}" \
      > "${RESULTS_DIR}/sgl_http_server.${SLURM_NODEID:-0}.log" 2>&1 &
  else
    log "SGL HTTP server script not set or missing. Skipping. Value='${script_path}'"
  fi
}

function launch_decode()
{
  wait_for_etcd

  local workers_per_node=${dynamo_args["decode-workers-per-node"]}
  local tp_size=${decode_args["--${dynamo_args["tp-arg-name"]}"]}
  local base_nixl_port=${VLLM_NIXL_SIDE_CHANNEL_PORT:-5557}
  local base_kv_event_port=${DYN_VLLM_KV_EVENT_PORT:-20080}
  log "Launching $workers_per_node decode worker(s) with unique port ranges"

  for i in $(seq 0 $(( $workers_per_node - 1 ))); do
    local gpu_list=$(_gpu_list_for_worker "${dynamo_args["decode-gpus-per-worker"]}" "$i")
    local log_file=$(_log_file_for_worker "decode" "$i")
    # Each worker needs unique port ranges to avoid ZMQ conflicts:
    # - NIXL side channel: base_port + (worker_index * tp_size) for TP ranks
    # - KV event port: one per worker
    local nixl_port=$((base_nixl_port + (i * tp_size)))
    local kv_event_port=$((base_kv_event_port + i))

    log "Launching decode worker $i on GPUs $gpu_list (NIXL port: $nixl_port, KV event port: $kv_event_port)"
    log "Decode cmd: ${dynamo_args["decode-cmd"]} $(array_to_args decode_args) ${decode_args["--extra-args"]}"
    CUDA_VISIBLE_DEVICES=$gpu_list \
      VLLM_NIXL_SIDE_CHANNEL_PORT=$nixl_port \
      DYN_VLLM_KV_EVENT_PORT=$kv_event_port \
      ${dynamo_args["decode-cmd"]} \
      $(array_to_args decode_args) ${decode_args["--extra-args"]} > $log_file 2>&1 &
  done
}

function wait_for_etcd()
{
  while [ "`curl -ks ${ETCD_ENDPOINTS}/readyz`" != "ok" ]; do
    log "Waiting for etcd to be ready by polling ${ETCD_ENDPOINTS}/readyz";
    sleep 10;
  done
  log "etcd is ready"
}

function launch_prefill()
{
  wait_for_etcd

  local workers_per_node=${dynamo_args["prefill-workers-per-node"]}
  local tp_size=${prefill_args["--${dynamo_args["tp-arg-name"]}"]}
  local base_nixl_port=${VLLM_NIXL_SIDE_CHANNEL_PORT:-5557}
  local base_kv_event_port=${DYN_VLLM_KV_EVENT_PORT:-20080}
  log "Launching $workers_per_node prefill worker(s) with unique port ranges"

  for i in $(seq 0 $(( $workers_per_node - 1 ))); do
    local gpu_list=$(_gpu_list_for_worker "${dynamo_args["prefill-gpus-per-worker"]}" "$i")
    local log_file=$(_log_file_for_worker "prefill" "$i")
    # Each worker needs unique port ranges to avoid ZMQ conflicts:
    # - NIXL side channel: base_port + (worker_index * tp_size) for TP ranks
    # - KV event port: one per worker
    local nixl_port=$((base_nixl_port + (i * tp_size)))
    local kv_event_port=$((base_kv_event_port + i))

    log "Launching prefill worker $i on GPUs $gpu_list (NIXL port: $nixl_port, KV event port: $kv_event_port)"
    log "Prefill cmd: ${dynamo_args["prefill-cmd"]} $(array_to_args prefill_args) ${prefill_args["--extra-args"]}"
    CUDA_VISIBLE_DEVICES=$gpu_list \
      VLLM_NIXL_SIDE_CHANNEL_PORT=$nixl_port \
      DYN_VLLM_KV_EVENT_PORT=$kv_event_port \
      ${dynamo_args["prefill-cmd"]} \
      $(array_to_args prefill_args) ${prefill_args["--extra-args"]} > $log_file 2>&1 &
  done
}

function launch_lmcache_controller()
{
  if [[ "$ENABLE_LMCACHE" != "1" ]]; then
    return
  fi

  log "Launching LMCache controller with cmd: ${dynamo_args["lmcache-controller-cmd"]}"
  ${dynamo_args["lmcache-controller-cmd"]} > ${RESULTS_DIR}/lmcache_controller.log 2>&1
}

function clear_lmcache()
{
  log "Clearing LMCache"

  response=$(curl -X POST http://${lmcache_config["controller_url"]}/clear \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "lmcache_default_instance",
    "location": "LocalCPUBackend"
  }')

  log "LMCache cleared. Response: $response"
}

function wait_for_dynamo_frontend()
{
  local want_prefill=0 #$(_expected_ready_prefill)
  local want_decode=$(_expected_ready_decode)

  while :; do
    local have_prefill=0 #$(_count_initialized_prefill)
    local have_decode=$(_count_initialized_decode)

    log "Initialized: prefill ${have_prefill}/${want_prefill}; decode ${have_decode}/${want_decode}"

    if [[ $have_prefill -ge $want_prefill && $have_decode -ge $want_decode ]]; then
      break
    fi

    exit_on_error
    sleep 30
  done

  log "Dynamo frontend is ready"
}

_probe_frontend_once() {
  local json='{
    "model": "'${dynamo_args["model"]}'",
    "messages": [{"role": "user", "content": "The color of sky is"}],
    "stream": false,
    "max_tokens": 10
  }'
  curl -s -X POST "${dynamo_args["url"]}/v1/chat/completions" -H "Content-Type: application/json" -d "$json"
}

function launch_genai_perf()
{
  wait_for_dynamo_frontend

  local genai_perf_arguments=$(array_to_args genai_perf_args)
  log "Launching genai-perf with cmd: ${dynamo_args["genai-perf-cmd"]} $genai_perf_arguments ${genai_perf_args["--extra-args"]}"

  ${genai_perf_wrapper_script} \
    --result_dir $RESULTS_DIR \
    --report_file ${genai_perf_report_file} \
    --calc_percentile_csv_path ${calc_percentile_csv_script} \
    --gpus_per_node $(_gpus_per_node) \
    -- ${dynamo_args["genai-perf-cmd"]} $genai_perf_arguments ${genai_perf_args["--extra-args"]} > ${RESULTS_DIR}/genai_perf.log 2>&1

  log "Done with genai-perf run"

  touch "$DONE_MARKER"
}

function setup_cufile()
{
  export CUFILE_ENV_PATH_JSON="$RESULTS_DIR/cufile.json"
  cat <<EOF > $CUFILE_ENV_PATH_JSON
{
    // NOTE : Application can override custom configuration via export CUFILE_ENV_PATH_JSON=<filepath>
    // e.g : export CUFILE_ENV_PATH_JSON="/home/<xxx>/cufile.json"
            "properties": {
                            // allow compat mode, this will enable use of cuFile posix read/writes
                            "allow_compat_mode": true,
                            // max IO chunk size (parameter should be multiples of 64K) used by cuFileRead/Write internally per IO request
                            "max_direct_io_size_kb" : 16384,
                            // device memory size (parameter should be 4K aligned) for reserving bounce buffers for the entire GPU
                            "max_device_cache_size_kb" : 2097152,
                            // Note: ensure (max_device_cache_size_kb / per_buffer_cache_size_kb) >= io_batchsize
                            // per-io bounce-buffer size (parameter should be multiples of 64K) ranging from 1024kb to 16384kb
                            "per_buffer_cache_size_kb": 16384,
                            // limit on maximum device memory size (parameter should be 4K aligned) that can be pinned for a given process
                            "max_device_pinned_mem_size_kb" : 33554432,

                            // posix bounce buffer pool size allocations
                            "posix_pool_slab_size_kb" : [16384],
                            // posix bounce buffer pool max counts
                            "posix_pool_slab_count": [1024]
            },
  "logging": {
    "dir": "$RESULTS_DIR",
    "level": "${CUFILE_LOG_LEVEL:-info}"
  }
}
EOF
}


function setup_kvbm()
{
  if [[ "$ENABLE_KVBM" != "1" ]]; then
    return
  fi

  if [[ -z "${DYN_KVBM_DISK_CACHE_DIR}" ]]; then
    log "ERROR: DYN_KVBM_DISK_CACHE_DIR is not set"
    exit 1
  fi

  rm -rf ${DYN_KVBM_DISK_CACHE_DIR}
  mkdir -p ${DYN_KVBM_DISK_CACHE_DIR}
  chmod 755 ${DYN_KVBM_DISK_CACHE_DIR}

  setup_cufile
}

function setup_lmcache()
{
  if [[ "$ENABLE_LMCACHE" != "1" ]]; then
    return
  fi

  local lmcache_path="${dynamo_args["lmcache-path"]}"
  log "Installing LMCache using: uv pip install $lmcache_path"
  uv pip install -e $lmcache_path

  local storage_cachedir="${dynamo_args["storage-cache-dir"]}/${dynamo_args["frontend-node"]}/"
  if [[ ${dynamo_args["clear-storage-cache-dir"]} == "true" ]]; then
    rm -rf $storage_cachedir 2>/dev/null || true
    mkdir -p $storage_cachedir
  fi

  export LMCACHE_CONFIG_FILE=$RESULTS_DIR/lmcache-nixl-config.yaml

  rm -f $LMCACHE_CONFIG_FILE

  for key in "${!lmcache_args[@]}"; do
    shopt -s nocasematch
    if [[ "$key" == "extra_config"* ]]; then
      continue
    fi

    val="${lmcache_args[$key]}"
    echo "$key: $val" >> $LMCACHE_CONFIG_FILE
  done

  echo "extra_config:" >> $LMCACHE_CONFIG_FILE
  for key in "${!lmcache_args[@]}"; do
    shopt -s nocasematch
    if [[ "$key" == "extra_config"* ]]; then
      nkey="${key#extra_config_}"
      val="${lmcache_args[$key]}"
      val=${val//%CACHEDIR%/${storage_cachedir}}
      echo "    $nkey: $val" >> $LMCACHE_CONFIG_FILE
    fi
  done
  setup_cufile
}

function launch_single_shot()
{
  wait_for_dynamo_frontend
  local isl="${dynamo_args["isl"]}"
  local lmcache_path="${dynamo_args["lmcache-path"]}"
  local url="${dynamo_args["url"]}"
  local cache_hit_pct="${dynamo_args["cache-hit-pct"]:-1}"

  local max_ctx_tokens_following=$(( $isl * 100 / $cache_hit_pct ))

  log "Launching single shot with lmcache path: $lmcache_path"
  log "python $lmcache_path/examples/online_session/openai_chat_completion_client.py --model ${dynamo_args["model"]} --api_base $url/v1 --max_ctx_tokens 131072 --num_following 1 "

  pushd $lmcache_path/examples/online_session
  log "python $lmcache_path/examples/online_session/openai_chat_completion_client.py --model ${dynamo_args["model"]} --api_base $url/v1 --max_ctx_tokens ${dynamo_args["isl"]} --context_file $lmcache_path/examples/online_session/salt.7.txt --out $RESULTS_DIR/single_shot.jsonl --num_following 1"

  python $lmcache_path/examples/online_session/openai_chat_completion_client.py \
    --model ${dynamo_args["model"]} \
    --api_base $url/v1 \
    --max_ctx_tokens $isl \
    --flush_cache \
    --context_file $lmcache_path/examples/online_session/salt.7.txt \
    --out $RESULTS_DIR/single_shot.jsonl \
    --num_following 1 > $RESULTS_DIR/single_shot_first_run.log 2>&1

    # --osl 10 \
    # --max_ctx_tokens_following ${max_ctx_tokens_following} \
  python -c "import pandas as pd; pd.read_json('$RESULTS_DIR/single_shot.jsonl', lines=True).to_csv('$RESULTS_DIR/report.csv', float_format='%.3f',index=False)"

  popd

  touch "$DONE_MARKER"
}

function launch_lmbench()
{
  wait_for_dynamo_frontend

  # run LMBenchmark, adjust the model name if you are using a different model
  # for detail how to config and run LMBenchmark: https://github.com/LMCache/LMBenchmark/tree/main/synthetic-multi-round-qa
  local lmbench_dir="${dynamo_args["lmbench-dir"]}"
  local log_file="${RESULTS_DIR}/lmbench.log"
  
  cmd="${dynamo_args["lmbench-cmd"]}"
  cmd=$(replace_placeholders "$cmd")
  cmd=${cmd//%LMBENCH_DIR%/${lmbench_dir}}

  pushd $RESULTS_DIR
  local lmbench_arguments=$(array_to_args lmbench_args)
  log "Launching lmbench with args: $cmd $lmbench_arguments ${lmbench_args["--extra-args"]}"

  $cmd ${lmbench_arguments} ${lmbench_args["--extra-args"]} > ${log_file} 2>&1

  log "Done with lmbench run"

  log "Summarizing lmbench run"
  python3 ${calc_percentile_csv_script} $RESULTS_DIR/lmcache_bench_output.csv -o $RESULTS_DIR/report.csv

  touch "$DONE_MARKER"
}

function wait_for_frontend_marker()
{
  while [ ! -f "$DONE_MARKER" ]; do
    exit_on_error
    log "Waiting for frontend completion marker by polling $DONE_MARKER"
    sleep 30
  done

  log "Done marker found."
}

function log_gpu_utilization()
{
  # Check if nvidia-smi is available
  if ! command -v nvidia-smi &> /dev/null; then
    log "Error: nvidia-smi not found"
    return
  fi

  wait_for_dynamo_frontend
  log "Starting GPU utilization monitoring"

  nvidia-smi \
    --query-gpu=timestamp,name,pci.bus_id,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used \
    --format=csv \
    -l 5 \
    -f ${RESULTS_DIR}/gpu_utilization-${SLURM_NODEID}.csv
}

function main()
{
  _init_runtime_env

  launch_node_setup_cmd

  validate_environment

  if _is_vllm; then
    cd ${dynamo_args["workspace-path"]}
  fi

  cd $RESULTS_DIR

  log_gpu_utilization &

  if _is_frontend_node; then
    log "Node ID: $SLURM_NODEID, Role: frontend"
    log_node_role "$(_current_node_name)" "frontend"
    setup_lmcache
    setup_kvbm
    launch_etcd &
    launch_nats &
    wait_for_etcd
    launch_ingress &
    if _is_sglang; then
      launch_sgl_http_server
    fi
  fi

  if _is_decode_node; then
    log "Node ID: $SLURM_NODEID, Role: decode"
    log_node_role "$(_current_node_name)" "decode"
    launch_decode &
  fi

  if _is_prefill_node; then
    log "Node ID: $SLURM_NODEID, Role: prefill"
    log_node_role "$(_current_node_name)" "prefill"
    #launch_prefill &
  fi

  if _is_frontend_node; then
    launch_lmcache_controller &

    if _is_genai_perf_workload; then
      launch_genai_perf &
    fi
    if _is_lmbench_workload; then
      launch_lmbench &
    fi
    if _is_single_shot_workload; then
      launch_single_shot &
    fi
  fi

  wait_for_frontend_marker
}

parse_args "$@"

log "env: $(env)"

log "Starting main"
main
log "Done with main"

perform_exit 0