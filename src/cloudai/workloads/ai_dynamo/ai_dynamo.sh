#!/bin/bash

# CloudAI params
RESULTS_DIR="/cloudai_run_results"
INSTALL_DIR="/cloudai_install"
STORAGE_CACHE_DIR="/cloudai_install/storage_cache"
HUGGINGFACE_HOME="/root/.cache/huggingface"
DONE_MARKER="./success-marker.txt"
FATAL_ERROR_MARKER="./failure-marker.txt"
NODE_ROLES_FILE="node_roles.log"
TEST_USER="$USER"

export DYN_SDK_DISABLE_ANSI_LOGGING=1
export VLLM_DISABLE_COLORED_OUTPUT=1
export VLLM_NO_COLOR=1
export VLLM_LOGGING_COLOR=0
#export VLLM_LOGGING_CONFIG_PATH="/cloudai_install/vllm_logging_config.json"

export ABSL_LOGGING_USE_COLOR=0
export DYN_LOGGING_DISABLE_ANSI_COLORS=1

export TERM=dumb
export NO_COLOR=1
export TQDM_DISABLE=1  # Disables tqdm progress bars globally
export TQDM_MININTERVAL=999999  # Makes tqdm update very rarely

export DEBIAN_FRONTEND=noninteractive
export APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

declare -A prefill_config
declare -A prefill_args
declare -A decode_config
declare -A decode_args
declare -A lmcache_args
declare -A lmcache_config
declare -A genai_perf_args
declare -A genai_perf_config
declare -A aiperf_args
declare -A aiperf_config
declare -A lmbench_args
declare -A lmbench_config
declare -A kvstorage_args
declare -A kvstorage_config

declare -A dynamo_args
dynamo_args["backend"]="vllm"
dynamo_args["node-setup-cmd"]=""
dynamo_args["ingress-cmd"]="python -m dynamo.frontend --router-mode kv"
dynamo_args["port"]=$((8080 + SLURM_JOBID % 100))
dynamo_args["endpoint"]="v1/chat/completions"
dynamo_args["model"]="Qwen/Qwen3-0.6B"
dynamo_args["etcd-port"]=2379
dynamo_args["nats-port"]=4222
dynamo_args["workspace-path"]="/workspace"
dynamo_args["frontend-node"]=""

dynamo_args["etcd-cmd"]="etcd --log-level debug"
dynamo_args["nats-cmd"]="nats-server -js"
dynamo_args["worker-error-pattern"]="zmq.error.ZMQError:.Address.already.in.use|UCX.*ERROR|ERROR.core.run_engine_core:.EngineCore.failed.to.start|ERROR.multiproc_executor.worker_busy_loop:.WorkerProc.hit.an.exception|EngineDeadError|EngineCore.encountered.an.issue"

# sglang-specific optional ports. Ignored by vllm.
dynamo_args["sgl-http-port"]=9001
dynamo_args["prefill-port"]=30011
dynamo_args["decode-port"]=30021


function log()
{
  echo -e "[$(date +%F\ %T) $(hostname)]: $*"
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
    if [[ "${arr[$i]}" == "$name" ]]; then
      echo "$i"; return 0
    fi
  done
  echo "-1"
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
  local prefill_nodes="${prefill_config["num-nodes"]}"
  if [[ "$prefill_nodes" -gt 0 ]]; then
    local prefill_master_host="$(_first_in_csv "${prefill_config["node-list"]}")"
    local prefill_master_ip="$(_resolve_host_ip "${prefill_master_host}")"
    local prefill_rank="$(_csv_index_of "${prefill_config["node-list"]}" "$self")"
    local prefill_total_gpus=$(( gpn * prefill_nodes ))
    prefill_args["--dist-init-addr"]="${prefill_master_ip}:${dynamo_args["prefill-port"]}"
    prefill_args["--nnodes"]="${prefill_nodes}"
    prefill_args["--node-rank"]="$([[ "$prefill_rank" -ge 0 ]] && echo "$prefill_rank" || echo 0)"
    prefill_args["--tp-size"]="${prefill_args["--tp-size"]:-${prefill_total_gpus}}"
    prefill_args["--dp-size"]="${prefill_args["--dp-size"]:-${prefill_total_gpus}}"
  fi

  # decode group
  local decode_nodes="${decode_config["num-nodes"]}"
  local decode_master_host="$(_first_in_csv "${decode_config["node-list"]}")"
  local decode_master_ip="$(_resolve_host_ip "${decode_master_host}")"
  local decode_rank="$(_csv_index_of "${decode_config["node-list"]}" "$self")"
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

_parse_cli_pairs() {
  log "Parsing args:"
  while [[ $# -ge 2 ]]; do
    echo "  $1 $2"
    key="$1"
    case $key in
      --workloads)
        dynamo_args["workloads"]="$2" ;;
      --dynamo-*)
        dynamo_args["${key#--dynamo-}"]="$2" ;;
      --workloads)
        dynamo_args["workloads"]="$2" ;;
      --prefill-args-*)
        prefill_args["--${key#--prefill-args-}"]="$2" ;;
      --prefill-*)
        prefill_config["${key#--prefill-}"]="$2" ;;
      --decode-args-*)
        decode_args["--${key#--decode-args-}"]="$2" ;;
      --decode-*)
        decode_config["${key#--decode-}"]="$2" ;;
      --lmcache-args-*)
        lmcache_args["${key#--lmcache-args-}"]="$2" ;;
      --lmcache-*)
        lmcache_config["${key#--lmcache-}"]="$2" ;;
      --lmbench-args-*)
        lmbench_args["--${key#--lmbench-args-}"]="$2" ;;
      --lmbench-*)
        lmbench_config["--${key#--lmbench-}"]="$2" ;;
      --genai_perf-args-*)
        genai_perf_args["--${key#--genai_perf-args-}"]="$2" ;;
      --genai_perf-*)
        genai_perf_config["--${key#--genai_perf-}"]="$2" ;;
      --aiperf-args-*)
        aiperf_args["--${key#--aiperf-args-}"]="$2" ;;
      --aiperf-*)
        aiperf_config["--${key#--aiperf-}"]="$2" ;;
      --kvstorage-args-*)
        kvstorage_args["--${key#--kvstorage-args-}"]="$2" ;;
      --kvstorage-*)
        kvstorage_config["--${key#--kvstorage-}"]="$2" ;;
      --hf-home)
        HUGGINGFACE_HOME="$2" ;;
      --storage-cache-dir)
        STORAGE_CACHE_DIR="$2" ;;
      --results-dir)
        RESULTS_DIR="$2" ;;
      --install-dir)
        INSTALL_DIR="$2" ;;
      --user)
        TEST_USER="$2" ;;
      --failure-marker)
        FATAL_ERROR_MARKER="$2" ;;
      --success-marker)
        DONE_MARKER="$2" ;;
    esac
    shift; shift;
  done
}

_populate_nodelist() {
  local num_nodes="$1"
  local exclude_nodelist="$2" 

  # Handle zero nodes case
  if [[ -z "$num_nodes" || "$num_nodes" -eq 0 ]]; then
    echo ""
    return
  fi

  local count=0
  local nodelist=""
  for node in $(echo "$DYNAMO_NODELIST" | tr ',' ' '); do
    if [[ -z "$node" ]]; then continue; fi
    if ! echo ",${exclude_nodelist}," | grep -q ",$node,"; then
      nodelist+="$node,"
      count=$(( count + 1 ))
      if [[ "$count" -eq "${num_nodes}" ]]; then
        break
      fi
    fi
  done

  # Terminate trailing comma
  nodelist=${nodelist%,}
  echo "$nodelist"
}

_set_nodelists()
{
  if [[ -z "${DYNAMO_NODELIST:-}" ]]; then
    log "ERROR: DYNAMO_NODELIST is not set"
    exit 1
  fi

  if [[ -z "${decode_config["node-list"]}" ]]; then
    decode_config["node-list"]=$(_populate_nodelist "${decode_config["num-nodes"]}" "")
  fi

  if [[ -z "${prefill_config["node-list"]}" ]]; then
    prefill_config["node-list"]=$(_populate_nodelist "${prefill_config["num-nodes"]}" "${decode_config["node-list"]}")
  fi

  # Prefill nodelist should match prefill node count (skip validation if num-nodes is 0)
  local prefill_num_nodes="${prefill_config["num-nodes"]:-0}"
  if [[ "$prefill_num_nodes" -gt 0 ]]; then
    local prefill_nodelist_count=$(_csv_len "${prefill_config["node-list"]}")
    if [[ "${prefill_nodelist_count}" -ne "${prefill_num_nodes}" ]]; then
      log "ERROR: number of nodes in prefill nodelist (${prefill_nodelist_count}) does not match prefill node count (${prefill_num_nodes})"
      exit 1
    fi
  fi

  local decode_nodelist_count=$(_csv_len "${decode_config["node-list"]}")
  if [[ "${decode_nodelist_count}" -ne "${decode_config["num-nodes"]}" ]]; then
    log "ERROR: number of nodes in decode nodelist (${decode_nodelist_count}) does not match decode node count (${decode_config["num-nodes"]})"
    exit 1
  fi
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

_has_connector() {
  # Check if a specific connector is in the comma-separated connector list.
  local needle="$1"
  local prefill_connectors="${prefill_args["--connector"]:-}"
  local decode_connectors="${decode_args["--connector"]:-}"
  [[ ",$prefill_connectors," == *",$needle,"* ]] || [[ ",$decode_connectors," == *",$needle,"* ]]
}

_apply_connector_settings() {
  if _has_connector "lmcache"; then
    ENABLE_LMCACHE=1
  fi
  if _has_connector "kvbm"; then
    ENABLE_KVBM=1
  fi
  if _has_connector "nixl"; then
    log "INFO: NIXL specified in the connector list"
  fi
}

_patch_dynamo_args() {
  if [[ -z "${dynamo_args["frontend-node"]}" ]]; then
    dynamo_args["frontend-node"]=$(echo "${decode_config["node-list"]}" | cut -d',' -f1)
  fi

  dynamo_args["url"]="http://${dynamo_args["frontend-node"]}:${dynamo_args["port"]}"
}

_patch_section_args() {
  if _is_sglang; then
    _apply_sglang_section_args
  fi
}

_compute_worker_allocation_sglang() {
  local num_gpus="$(_gpus_per_node)"
  if [[ $num_gpus -eq 0 ]]; then
    log "ERROR: No GPUs found in CUDA_VISIBLE_DEVICES"
    exit 1
  fi

  # sglang: one worker per node using all GPUs
  prefill_config["gpus-per-worker"]=$num_gpus
  decode_config["gpus-per-worker"]=$num_gpus
  prefill_config["workers-per-node"]=1
  decode_config["workers-per-node"]=1
}

_compute_worker_allocation_vllm() {
  local num_gpus="$(_gpus_per_node)"

  if [[ $num_gpus -eq 0 ]]; then
    log "ERROR: No GPUs found in CUDA_VISIBLE_DEVICES"
    exit 1
  fi

  prefill_config["gpus-per-worker"]=$(( prefill_args["--tensor-parallel-size"] * prefill_args["--pipeline-parallel-size"] ))
  decode_config["gpus-per-worker"]=$(( decode_args["--tensor-parallel-size"] * decode_args["--pipeline-parallel-size"] ))

  if [[ ${prefill_config["gpus-per-worker"]} -eq 0 ]] || [[ ${decode_config["gpus-per-worker"]} -eq 0 ]]; then
    log "ERROR: Invalid TP/PP configuration"
    exit 1
  fi

  if [[ "${prefill_config["multiple-workers-per-node"]}" != "true" ]]; then
    prefill_config["gpus-per-worker"]=$num_gpus
  fi

  if [[ "${decode_config["multiple-workers-per-node"]}" != "true" ]]; then
    decode_config["gpus-per-worker"]=$num_gpus
  fi

  log "DECODE: num GPUs: $num_gpus, GPUs per worker: ${decode_config["gpus-per-worker"]}"
  log "PREFILL: num GPUs: $num_gpus, GPUs per worker: ${prefill_config["gpus-per-worker"]}"
  prefill_config["workers-per-node"]=$(( num_gpus / prefill_config["gpus-per-worker"] ))
  decode_config["workers-per-node"]=$(( num_gpus / decode_config["gpus-per-worker"] ))
  log "DECODE: workers per node: ${decode_config["workers-per-node"]}"
  log "PREFILL: workers per node: ${prefill_config["workers-per-node"]}"

  log "NUM PREFILL NODES: ${prefill_config["num-nodes"]}"
  log "NUM DECODE NODES: ${decode_config["num-nodes"]}"
}

_compute_worker_allocation() {
  if _is_sglang; then
    _compute_worker_allocation_sglang
  else
    _compute_worker_allocation_vllm
  fi
}

arg_array_to_string()
{
  local -n arr=$1
  local result=""
  for key in "${!arr[@]}"; do
    result+="    ${key} ${arr[$key]}\n"
  done
  echo -e "$result"
}

_dump_args() {
  log "Dynamo args:\n$(arg_array_to_string dynamo_args)"
  log "Prefill config params:\n$(arg_array_to_string prefill_config)"
  log "Prefill args:\n$(arg_array_to_string prefill_args)"
  log "Decode config params:\n$(arg_array_to_string decode_config)"
  log "Decode args:\n$(arg_array_to_string decode_args)"
  log "LMCache config params:\n$(arg_array_to_string lmcache_config)"
  log "LMCache args:\n$(arg_array_to_string lmcache_args)"
  log "GenAI config params:\n$(arg_array_to_string genai_perf_config)"
  log "GenAI-Perf args:\n$(arg_array_to_string genai_perf_args)"
  log "AIPerf config params:\n$(arg_array_to_string aiperf_config)"
  log "AIPerf args:\n$(arg_array_to_string aiperf_args)"
  log "LMBench config params:\n$(arg_array_to_string lmbench_config)"
  log "LMBench args:\n$(arg_array_to_string lmbench_args)"
  log "KV storage config params:\n$(arg_array_to_string kvstorage_config)"
  log "KV storage args:\n$(arg_array_to_string kvstorage_args)"
  log "--------------------------------"
}

function parse_args()
{
  _parse_cli_pairs "$@"
  _set_nodelists
  _set_backend_defaults
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
  val=${val//%INSTALL_DIR%/${INSTALL_DIR}}
  val=${val//%HUGGINGFACE_HOME%/${HUGGINGFACE_HOME}}
  echo "$val"
}

function array_to_args()
{
  local -n arr=$1
  local result=""
  for key in "${!arr[@]}"; do
    shopt -s nocasematch
    val=$(replace_placeholders "${arr[$key]}")
    # Quote values that contain spaces
    if [[ "$val" == *" "* ]]; then
      val="${val//\"/\\\"}"  # Escape existing quotes
      result+="${key} \"${val}\" "
    else
      result+="${key} ${val} "
    fi
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
  echo $(( prefill_config["num-nodes"] * prefill_config["workers-per-node"] ))
}

_total_workers_decode() {
  echo $(( decode_config["num-nodes"] * decode_config["workers-per-node"] ))
}

_count_initialized_prefill() {
  grep -i -l -E "${prefill_config["worker-initialized-regex"]}" "${RESULTS_DIR}"/dynamo_*prefill* 2>/dev/null | wc -l
}

_count_initialized_decode() {
  grep -i -l -E "${decode_config["worker-initialized-regex"]}" "${RESULTS_DIR}"/dynamo_*decode* 2>/dev/null | wc -l
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
  [[ ",${dynamo_args["frontend-node"]}," == *",$name,"* ]]
}

_is_decode_node() {
  local name="$(_current_node_name)"
  [[ ",${decode_config["node-list"]}," == *",$name,"* ]]
}

_is_prefill_node() {
  local name="$(_current_node_name)"
  [[ ",${prefill_config["node-list"]}," == *",$name,"* ]]
}

_is_genai_perf_workload() {
  [[ "${dynamo_args["workloads"]}" == *"genai_perf.sh"* ]]
}

_is_aiperf_workload() {
  [[ "${dynamo_args["workloads"]}" == *"aiperf.sh"* ]]
}

_is_lmbench_workload() {
  [[ "${dynamo_args["workloads"]}" == *"lmbench.sh"* ]]
}

_is_kvstorage_workload() {
  [[ "${dynamo_args["workloads"]}" == *"kvstorage.sh"* ]]
}

_init_runtime_env() {
  if _is_vllm; then
    export HF_HOME="${HUGGINGFACE_HOME}"
    hf cache scan
  fi
  export NATS_SERVER="nats://${dynamo_args["frontend-node"]}:${dynamo_args["nats-port"]}"
  export ETCD_ENDPOINTS="http://${dynamo_args["frontend-node"]}:${dynamo_args["etcd-port"]}"
  export UCX_LOG_FILE="${RESULTS_DIR}/ucx_log_%h.log"
}

function launch_node_setup_cmd()
{
  logfile="${RESULTS_DIR}/node_setup_$(_current_node_name).log"
  if [[ -n "${dynamo_args["node-setup-cmd"]}" ]]; then
    log "Launching node setup command: ${dynamo_args["node-setup-cmd"]}"
    bash -c "${dynamo_args["node-setup-cmd"]}" >> "$logfile" 2>&1
    log "Node setup complete"
  fi

  log "Node environment:\n$(env)" >> "$logfile" 2>&1
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
  _require_cmd jq
  _require_cmd uv

  # Runtime commands invoked later
  _require_cmd python3
  _require_cmd ${dynamo_args["etcd-cmd"]%% *}     # first token if args included
  _require_cmd ${dynamo_args["nats-cmd"]%% *}

  # Basic env presence
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    log "ERROR: CUDA_VISIBLE_DEVICES is not set"
    exit 1
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

function wait_for_frontend_marker()
{
  while [ ! -f "$DONE_MARKER" ]; do
    exit_on_error
    log "Waiting for frontend completion marker by polling $DONE_MARKER"
    sleep 30
  done

  log "Done marker found."
}

function mark_done()
{
  touch "$DONE_MARKER"
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

  local workers_per_node=${decode_config["workers-per-node"]}
  local tp_size=${decode_args["--tensor-parallel-size"]}
  local base_nixl_port=${VLLM_NIXL_SIDE_CHANNEL_PORT:-5557}
  local base_kv_event_port=${DYN_VLLM_KV_EVENT_PORT:-20080}
  log "Launching $workers_per_node decode worker(s) with unique port ranges"

  for i in $(seq 0 $(( $workers_per_node - 1 ))); do
    local gpu_list=$(_gpu_list_for_worker "${decode_config["gpus-per-worker"]}" "$i")
    local log_file=$(_log_file_for_worker "decode" "$i")
    # Each worker needs unique port ranges to avoid ZMQ conflicts:
    # - NIXL side channel: base_port + (worker_index * tp_size) for TP ranks
    # - KV event port: one per worker
    local nixl_port=$((base_nixl_port + (i * tp_size)))
    local kv_event_port=$((base_kv_event_port + i))

    # Build decode args as proper bash arrays to preserve
    # multi-word values (e.g. --cmd "aiperf profile") through word splitting.
    local -a args_arr=()
    for key in "${!decode_args[@]}"; do
      args_arr+=($key $(replace_placeholders "${decode_args[$key]}"))
    done

    log "Launching decode worker $i on GPUs $gpu_list (NIXL port: $nixl_port, KV event port: $kv_event_port)"
    log "Decode cmd: ${decode_config["cmd"]} ${args_arr[*]} ${decode_config["extra-args"]}"
    CUDA_VISIBLE_DEVICES=$gpu_list \
      VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname -I | awk '{print $1}') \
      VLLM_NIXL_SIDE_CHANNEL_PORT=$nixl_port \
      DYN_VLLM_KV_EVENT_PORT=$kv_event_port \
      ${decode_config["cmd"]} \
      ${args_arr[@]} \
      ${decode_config["extra-args"]} > $log_file 2>&1 &
  done
}

function wait_for_etcd()
{
  while [ "$(curl -ks ${ETCD_ENDPOINTS}/readyz)" != "ok" ]; do
    log "Waiting for etcd to be ready by polling ${ETCD_ENDPOINTS}/readyz";
    sleep 10;
  done
  log "etcd is ready"
}

function launch_prefill()
{
  wait_for_etcd

  local workers_per_node=${prefill_config["workers-per-node"]}
  local tp_size=${prefill_args["--tensor-parallel-size"]}
  local base_nixl_port=${VLLM_NIXL_SIDE_CHANNEL_PORT:-5557}
  local base_kv_event_port=${DYN_VLLM_KV_EVENT_PORT:-20080}
  log "Launching $workers_per_node prefill worker(s) with unique port ranges"

  for i in $(seq 0 $(( $workers_per_node - 1 ))); do
    local gpu_list=$(_gpu_list_for_worker "${prefill_config["gpus-per-worker"]}" "$i")
    local log_file=$(_log_file_for_worker "prefill" "$i")
    # Each worker needs unique port ranges to avoid ZMQ conflicts:
    # - NIXL side channel: base_port + (worker_index * tp_size) for TP ranks
    # - KV event port: one per worker
    local nixl_port=$((base_nixl_port + (i * tp_size)))
    local kv_event_port=$((base_kv_event_port + i))

    # Build prefill args as proper bash arrays to preserve
    # multi-word values (e.g. --cmd "aiperf profile") through word splitting.
    local -a args_arr=()
    for key in "${!prefill_args[@]}"; do
      args_arr+=($key $(replace_placeholders "${prefill_args[$key]}"))
    done

    log "Launching prefill worker $i on GPUs $gpu_list (NIXL port: $nixl_port, KV event port: $kv_event_port)"
    log "Prefill cmd: ${prefill_config["cmd"]} ${args_arr[*]} ${prefill_config["extra-args"]}"
    CUDA_VISIBLE_DEVICES=$gpu_list \
      VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname -I | awk '{print $1}') \
      VLLM_NIXL_SIDE_CHANNEL_PORT=$nixl_port \
      DYN_VLLM_KV_EVENT_PORT=$kv_event_port \
      ${prefill_config["cmd"]} \
      ${args_arr[@]} \
      ${prefill_config["extra-args"]} > $log_file 2>&1 &
  done
}

function launch_lmcache_controller()
{
  if ! _has_connector "lmcache"; then
    return
  fi

  log "Launching LMCache controller with cmd: ${lmcache_config["controller_cmd"]}"
  ${lmcache_config["controller_cmd"]} > ${RESULTS_DIR}/lmcache_controller.log 2>&1
}

function wait_for_dynamo_frontend()
{
  local want_prefill=$(_expected_ready_prefill)
  local want_decode=$(_expected_ready_decode)

  while :; do
    local have_prefill=$(_count_initialized_prefill)
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

_query_frontend() {
  local content="${1:-The color of sky is}"
  content=$(echo "$content" | sed 's/"/\\"/g' | sed 's/\n/\\n/g')
  local max_tokens="${2:-10}"

  local json='{
    "model": "'${dynamo_args["model"]}'",
    "messages": [{"role": "user", "content": "'"$content"'"}],
    "stream": false,
    "max_tokens": '$max_tokens',
    "temperature": 0,
    "top_p": 0.0001
  }'

  echo "$json" > "$RESULTS_DIR/curl_cmd.json"
  curl -s -X POST "${dynamo_args["url"]}/v1/chat/completions" -H "Content-Type: application/json" -d @$RESULTS_DIR/curl_cmd.json
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
    "level": "${CUFILE_LOG_LEVEL:-INFO}"
  }
}
EOF
}

function setup_storage_cache_dir()
{
  local connector="$1"
  # Use a global variable that can be exported
  STORAGE_CACHE_DIR="$STORAGE_CACHE_DIR/${TEST_USER}/${dynamo_args["frontend-node"]}/${connector}/cache"
  rm -rf "${STORAGE_CACHE_DIR}"
  mkdir -p "${STORAGE_CACHE_DIR}"
  chmod 755 "${STORAGE_CACHE_DIR}"
}

function setup_kvbm()
{
  if ! _has_connector "kvbm"; then
    log "Connector list does not include kvbm. Skipping setup_kvbm"
    return
  fi

  log "Setting up KVBM storage cache directory: ${STORAGE_CACHE_DIR}"
  setup_storage_cache_dir "kvbm"
  export DYN_KVBM_DISK_CACHE_DIR=${STORAGE_CACHE_DIR}
  setup_cufile
}

function setup_lmcache()
{
  if ! _has_connector "lmcache"; then
    log "Connector list does not include lmcache. Skipping setup_lmcache"
    return
  fi

  log "Setting up LMCache; installing LMCache using: uv pip install $lmcache_path"
  local lmcache_path="${lmcache_config["repo"]}"
  uv pip install -e $lmcache_path

  setup_storage_cache_dir "lmcache"

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
      val=${val//%CACHEDIR%/${STORAGE_CACHE_DIR}}
      echo "    $nkey: $val" >> $LMCACHE_CONFIG_FILE
    fi
  done
  setup_cufile
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

function launch_workload()
{
  local workload_config_name="$1"
  local workload_args_name="$2"

  # Create nameref to the associative arrays
  local -n workload_config_ref="$workload_config_name"
  local -n workload_args_ref="$workload_args_name"

  local workload_name="${workload_config_ref["--name"]}"
  local script="${workload_config_ref["--script"]}"

  # Build config and workload args as proper bash arrays to preserve
  # multi-word values (e.g. --cmd "aiperf profile") through word splitting.
  local -a config_arr=()
  for key in "${!workload_config_ref[@]}"; do
    config_arr+=("$key" "$(replace_placeholders "${workload_config_ref[$key]}")")
  done

  local -a args_arr=()
  for key in "${!workload_args_ref[@]}"; do
    args_arr+=("$key" "$(replace_placeholders "${workload_args_ref[$key]}")")
  done

  log "Launching $workload_name with cmd: ${INSTALL_DIR}/$script ${config_arr[*]} -- ${args_arr[*]}"

  bash "${INSTALL_DIR}/$script" \
    --install_dir "$INSTALL_DIR" \
    --result_dir "$RESULTS_DIR" \
    --model "${dynamo_args["model"]}" \
    --url "http://${dynamo_args["frontend-node"]}" \
    --port "${dynamo_args["port"]}" \
    --endpoint "${dynamo_args["endpoint"]}" \
    --gpus_per_node "$(_gpus_per_node)" \
    --decode-connector "${decode_args["--connector"]}" \
    --prefill-connector "${prefill_args["--connector"]}" \
    --kvbm_metrics_port "${DYN_KVBM_METRICS_PORT:-6880}" \
    --decode-nodes "${decode_config["node-list"]}" \
    "${config_arr[@]}" \
    -- "${args_arr[@]}" > "${RESULTS_DIR}/$workload_name.log" 2>&1

  log "Done with $workload_name run"
}

function launch_workloads()
{
  wait_for_dynamo_frontend

  if _is_genai_perf_workload; then
    launch_workload genai_perf_config genai_perf_args
  fi
  if _is_aiperf_workload; then
    launch_workload aiperf_config aiperf_args
  fi
  if _is_lmbench_workload; then
    launch_workload lmbench_config lmbench_args
  fi
  if _is_kvstorage_workload; then
    launch_workload kvstorage_config kvstorage_args
  fi

  mark_done
}

function main()
{
  parse_args "$@"

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
    launch_prefill &
  fi

  if _is_frontend_node; then
    launch_lmcache_controller &

    sleep 10

    launch_workloads &
  fi

  wait_for_frontend_marker
}

log "Starting main"
main "$@"
log "Done with main"

perform_exit 0
