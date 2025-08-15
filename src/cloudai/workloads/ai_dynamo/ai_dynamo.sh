#!/bin/bash

# CloudAI params
RESULTS_DIR="/cloudai_run_results"
HUGGINGFACE_HOME="/root/.cache/huggingface"
DONE_MARKER="frontend_done.marker"
NODE_ROLES_FILE="node_roles.log"

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
declare -A genai_perf_args

declare -A dynamo_args
dynamo_args["backend"]="vllm"
dynamo_args["node-setup-cmd"]=""
dynamo_args["prefill-cmd"]="python3 -m dynamo.vllm --is-prefill-worker"
dynamo_args["decode-cmd"]="python3 -m dynamo.vllm"
dynamo_args["ingress-cmd"]="python -m dynamo.frontend --router-mode kv"
dynamo_args["port"]=8080
dynamo_args["endpoint"]="v1/chat/completions"
dynamo_args["model"]="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
dynamo_args["etcd-cmd"]="etcd --log-level debug"
dynamo_args["nats-cmd"]="nats-server -js"
dynamo_args["etcd-port"]=2379
dynamo_args["nats-port"]=4222
dynamo_args["workspace-path"]="/workspace"
dynamo_args["frontend-node"]=""
dynamo_args["num-prefill-nodes"]=1
dynamo_args["num-decode-nodes"]=1
dynamo_args["prefill-nodelist"]=""
dynamo_args["decode-nodelist"]=""
dynamo_args["tp-arg-name"]="tensor-parallel-size"
dynamo_args["pp-arg-name"]="pipeline-parallel-size"
dynamo_args["multiple-prefill-workers-per-node"]="true"
dynamo_args["multiple-decode-workers-per-node"]="true"
dynamo_args["prefill-initialized-regex"]="prefill.*initialized"
dynamo_args["decode-initialized-regex"]="decode.*initialized"

# sglang-specific optional ports. Ignored by vllm.
dynamo_args["sgl-http-port"]=9001
dynamo_args["prefill-port"]=30011
dynamo_args["decode-port"]=30021

# GenAI Perf params
GENAI_PERF_PROFILE_EXPORT_FILE="profile.json"
GENAI_PERF_ARTIFACT_DIR="genai_perf_artifacts"

function log()
{
  echo "[$(date --iso-8601=ns) $(hostname)]: $@"
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
      --genai-perf-*)
        genai_perf_args["--${key#--genai-perf-}"]="$2" ;;
      --huggingface-home)
        HUGGINGFACE_HOME="$2" ;;
      --results-dir)
        RESULTS_DIR="$2" ;;
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

_is_vllm() { [[ "${dynamo_args["backend"]}" == "vllm" ]]; }
_is_sglang() { [[ "${dynamo_args["backend"]}" == "sglang" ]]; }

_sync_num_nodes_from_section_args() {
  if [[ -n "${prefill_args["--num-nodes"]:-}" ]]; then
    dynamo_args["num-prefill-nodes"]="${prefill_args["--num-nodes"]}"
  fi
  if [[ -n "${decode_args["--num-nodes"]:-}" ]]; then
    dynamo_args["num-decode-nodes"]="${decode_args["--num-nodes"]}"
  fi
}

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
  local dl_len=$(_csv_len "${dynamo_args["decode-nodelist"]}")
  local pl_len=$(_csv_len "${dynamo_args["prefill-nodelist"]}")
  if (( dl_len > 0 )); then dynamo_args["num-decode-nodes"]="$dl_len"; fi
  if (( pl_len > 0 )); then dynamo_args["num-prefill-nodes"]="$pl_len"; fi

  if [[ -z "${dynamo_args["decode-nodelist"]}" || -z "${dynamo_args["prefill-nodelist"]}" ]]; then
    if [[ -z "${DYNAMO_NODELIST:-}" ]]; then
      log "ERROR: Provide --dynamo-decode-nodelist/--dynamo-prefill-nodelist or set DYNAMO_NODELIST"; exit 1
    fi
    local d="${dynamo_args["num-decode-nodes"]}"
    local p="${dynamo_args["num-prefill-nodes"]}"
    local total=$(_csv_len "${DYNAMO_NODELIST}")
    if (( total < d + p )); then
      log "ERROR: DYNAMO_NODELIST has ${total} entries; need decode(${d})+prefill(${p})"; exit 1
    fi
    [[ -z "${dynamo_args["decode-nodelist"]}" ]] && \
      dynamo_args["decode-nodelist"]=$(echo "$DYNAMO_NODELIST" | cut -d',' -f1-"$d")
    [[ -z "${dynamo_args["prefill-nodelist"]}" ]] && \
      dynamo_args["prefill-nodelist"]=$(echo "$DYNAMO_NODELIST" | cut -d',' -f$(( d + 1 ))-)
  fi
}

_patch_dynamo_args() {
  if [[ -z "${dynamo_args["decode-nodelist"]}" ]]; then
    dynamo_args["decode-nodelist"]=$(echo $DYNAMO_NODELIST | cut -d',' -f1-${dynamo_args["num-decode-nodes"]})
  fi

  if [[ -z "${dynamo_args["prefill-nodelist"]}" ]]; then
    dynamo_args["prefill-nodelist"]=$(echo $DYNAMO_NODELIST | cut -d',' -f$(( ${dynamo_args["num-decode-nodes"]} + 1 ))-)
  fi

  if [[ -z "${dynamo_args["frontend-node"]}" ]]; then
    dynamo_args["frontend-node"]=$(echo ${dynamo_args["decode-nodelist"]} | cut -d',' -f1)
  fi

  dynamo_args["url"]="http://${dynamo_args["frontend-node"]}:${dynamo_args["port"]}"

  _validate_or_build_nodelists
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
  prefill_args["--port"]=${dynamo_args["prefill-port"]}
  decode_args["--port"]=${dynamo_args["decode-port"]}
  prefill_args["--served-model-name"]=${dynamo_args["model"]}
  decode_args["--served-model-name"]=${dynamo_args["model"]}

  # model-path must point to HF cache for sglang
  prefill_args["--model-path"]="${HUGGINGFACE_HOME}"
  decode_args["--model-path"]="${HUGGINGFACE_HOME}"

  local self="$(_current_node_name)"
  local gpn="$(_gpus_per_node)"

  # prefill group
  local prefill_nodes="${dynamo_args["num-prefill-nodes"]}"
  local prefill_master_host="$(_first_in_csv "${dynamo_args["prefill-nodelist"]}")"
  local prefill_master_ip="$(_resolve_host_ip "${prefill_master_host}")"
  local prefill_rank="$(_csv_index_of "${dynamo_args["prefill-nodelist"]}" "$self")"
  local prefill_total_gpus=$(( gpn * prefill_nodes ))
  prefill_args["--dist-init-addr"]="${prefill_master_ip}:${dynamo_args["prefill-port"]}"
  prefill_args["--nnodes"]="${prefill_nodes}"
  prefill_args["--node-rank"]="$([[ "$prefill_rank" -ge 0 ]] && echo "$prefill_rank" || echo 0)"
  prefill_args["--tp-size"]="${prefill_total_gpus}"
  prefill_args["--dp-size"]="${prefill_total_gpus}"

  # decode group
  local decode_nodes="${dynamo_args["num-decode-nodes"]}"
  local decode_master_host="$(_first_in_csv "${dynamo_args["decode-nodelist"]}")"
  local decode_master_ip="$(_resolve_host_ip "${decode_master_host}")"
  local decode_rank="$(_csv_index_of "${dynamo_args["decode-nodelist"]}" "$self")"
  local decode_total_gpus=$(( gpn * decode_nodes ))
  decode_args["--dist-init-addr"]="${decode_master_ip}:${dynamo_args["decode-port"]}"
  decode_args["--nnodes"]="${decode_nodes}"
  decode_args["--node-rank"]="$([[ "$decode_rank" -ge 0 ]] && echo "$decode_rank" || echo 0)"
  decode_args["--tp-size"]="${decode_total_gpus}"
  decode_args["--dp-size"]="${decode_total_gpus}"

  if [[ -n "${dynamo_args["deepep-config"]:-}" ]]; then
    [[ -f "${dynamo_args["deepep-config"]}" ]] || log "WARN: deepep-config not found: ${dynamo_args["deepep-config"]}"
    prefill_args["--deepep-config"]="${dynamo_args["deepep-config"]}"
    decode_args["--deepep-config"]="${dynamo_args["deepep-config"]}"
  fi

  unset 'prefill_args["--model"]'
  unset 'decode_args["--model"]'
}

_apply_genai_perf_section_args() {
  genai_perf_args["--model"]="${dynamo_args["model"]}"
  genai_perf_args["--url"]="${dynamo_args["url"]}"
  genai_perf_args["--endpoint"]="${dynamo_args["endpoint"]}"
  genai_perf_args["--artifact-dir"]="${RESULTS_DIR}/${GENAI_PERF_ARTIFACT_DIR}/"
  genai_perf_args["--profile-export-file"]="${GENAI_PERF_PROFILE_EXPORT_FILE}"
}

_patch_section_args() {
  prefill_args["--model"]="${dynamo_args["model"]}"
  decode_args["--model"]="${dynamo_args["model"]}"

  if _is_sglang; then
    _apply_sglang_section_args
  fi

  _apply_genai_perf_section_args
}

_compute_worker_allocation_sglang() {
  local num_gpus=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c .)
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
  local num_gpus=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c .)

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

  dynamo_args["prefill-workers-per-node"]=$(( num_gpus / dynamo_args["prefill-gpus-per-worker"] ))
  dynamo_args["decode-workers-per-node"]=$(( num_gpus / dynamo_args["decode-gpus-per-worker"] ))

  if [[ -n "${prefill_args["--num-nodes"]}" ]]; then
    dynamo_args["num-prefill-nodes"]=${prefill_args["--num-nodes"]}
  fi
  if [[ -n "${decode_args["--num-nodes"]}" ]]; then
    dynamo_args["num-decode-nodes"]=${decode_args["--num-nodes"]}
  fi
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
}

function parse_args()
{
  _parse_cli_pairs "$@"
  _set_backend_defaults
  _sync_num_nodes_from_section_args
  _patch_dynamo_args
  _patch_section_args
  _compute_worker_allocation
  _dump_args
}

function array_to_args()
{
  local -n arr=$1
  local result=""
  for key in "${!arr[@]}"; do
    if [[ "$key" == "--extra-args" ]]; then
      continue
    elif [[ "$key" == "--num-nodes" ]]; then
      continue
    else
      result+="${key} ${arr[$key]} "
    fi
  done
  echo "$result"
}

function launch_node_setup_cmd()
{
  if [[ -n "${dynamo_args["node-setup-cmd"]}" ]]; then
    log "Launching node setup command: ${dynamo_args["node-setup-cmd"]}"
    bash -c "${dynamo_args["node-setup-cmd"]}"
    log "Node setup complete"
  fi
}

function launch_etcd()
{
  log "Launching etcd"
  ${dynamo_args["etcd-cmd"]} \
    --listen-client-urls http://0.0.0.0:${dynamo_args["etcd-port"]} \
    --advertise-client-urls http://0.0.0.0:${dynamo_args["etcd-port"]} \
    > ${RESULTS_DIR}/etcd.log 2>&1
}

function launch_nats()
{
  log "Launching nats"
  ${dynamo_args["nats-cmd"]} -p ${dynamo_args["nats-port"]} > ${RESULTS_DIR}/nats.log 2>&1
}

function wait_for_etcd()
{
  while [ "`curl -ks ${ETCD_ENDPOINTS}/readyz`" != "ok" ]; do
    log "Waiting for etcd to be ready by polling ${ETCD_ENDPOINTS}/readyz";
    sleep 10;
  done
  log "etcd is ready"
}

function launch_ingress()
{
  log "Launching ingress with cmd: ${dynamo_args["ingress-cmd"]} --http-port ${dynamo_args["port"]}"
  ${dynamo_args["ingress-cmd"]} --http-port ${dynamo_args["port"]} > ${RESULTS_DIR}/dynamo_ingress.log 2>&1
}

function exit_on_error()
{
  num_failed_workers=$(grep "zmq.error.ZMQError: Address already in use" $RESULTS_DIR/dynamo_*.log -il 2> /dev/null |wc -l)
  if [[ $num_failed_workers -gt 0 ]]; then
    log "ZMQ ERROR: Found $num_failed_workers failed workers, exiting"
    log "Killing all jobs"
    kill $(jobs -p) 2> /dev/null
    exit 1
  fi

  num_failed_workers=$(grep "UCX.*ERROR" $RESULTS_DIR/ucx_log_*.log -il 2> /dev/null |wc -l)
  if [[ $num_failed_workers -gt 0 ]]; then
    log "UCX ERROR: Found $num_failed_workers failed workers, exiting"
    log "Killing all jobs"
    kill $(jobs -p)
    exit 1
  fi
}

_total_workers_prefill() {
  echo $(( dynamo_args["num-prefill-nodes"] * dynamo_args["prefill-workers-per-node"] ))
}

_total_workers_decode() {
  echo $(( dynamo_args["num-decode-nodes"] * dynamo_args["decode-workers-per-node"] ))
}

_count_initialized_prefill() {
  if _is_sglang; then
    grep -l "Request handler initialized" "${RESULTS_DIR}"/dynamo_*prefill* 2>/dev/null | wc -l
  else
    grep -i -l -E "${dynamo_args["prefill-initialized-regex"]}" "${RESULTS_DIR}"/dynamo_*prefill* 2>/dev/null | wc -l
  fi
}

_count_initialized_decode() {
  if _is_sglang; then
    grep -l "Decode request handler initialized" "${RESULTS_DIR}"/dynamo_*decode* 2>/dev/null | wc -l
  else
    grep -i -l -E "${dynamo_args["decode-initialized-regex"]}" "${RESULTS_DIR}"/dynamo_*decode* 2>/dev/null | wc -l
  fi
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

_probe_frontend_once() {
  local json='{
    "model": "'${dynamo_args["model"]}'",
    "messages": [{"role": "user", "content": "The color of sky is"}],
    "stream": false,
    "max_tokens": 10
  }'
  curl -s -X POST "${dynamo_args["url"]}/v1/chat/completions" -H "Content-Type: application/json" -d "$json"
}

function wait_for_dynamo_frontend()
{
  local want_prefill=$(_expected_ready_prefill)
  local want_decode=$(_expected_ready_decode)

  while :; do
    local have_prefill=$(_count_initialized_prefill)
    local have_decode=$(_count_initialized_decode)

    if [[ $have_prefill -ge $want_prefill && $have_decode -ge $want_decode ]]; then
      break
    fi

    log "Initialized: prefill ${have_prefill}/${want_prefill}; decode ${have_decode}/${want_decode}"
    exit_on_error
    sleep 30
  done

  log "Dynamo frontend is ready"
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

function launch_genai_perf()
{
  wait_for_dynamo_frontend

  local resp=$(_probe_frontend_once)
  echo "Response: $resp"

  local genai_perf_arguments=$(array_to_args genai_perf_args)
  log "Launching genai-perf with args: $genai_perf_arguments ${genai_perf_args["--extra-args"]}"

  ${dynamo_args["genai-perf-cmd"]} ${genai_perf_arguments} ${genai_perf_args["--extra-args"]} > ${RESULTS_DIR}/genai_perf.log 2>&1

  log "Done with genai-perf run"
}

function launch_prefill()
{
  wait_for_etcd

  local workers_per_node=${dynamo_args["prefill-workers-per-node"]}

  for i in $(seq 0 $(( $workers_per_node - 1 ))); do
    local gpu_list=$(_gpu_list_for_worker "${dynamo_args["prefill-gpus-per-worker"]}" "$i")
    local log_file=$(_log_file_for_worker "prefill" "$i")

    log "Launching prefill worker $i on GPUs $gpu_list"
    CUDA_VISIBLE_DEVICES=$gpu_list \
      ${dynamo_args["prefill-cmd"]} \
      $(array_to_args prefill_args) ${prefill_args["--extra-args"]} > $log_file 2>&1 &
  done
}

function launch_decode()
{
  wait_for_etcd

  local workers_per_node=${dynamo_args["decode-workers-per-node"]}

  for i in $(seq 0 $(( $workers_per_node - 1 ))); do
    local gpu_list=$(_gpu_list_for_worker "${dynamo_args["decode-gpus-per-worker"]}" "$i")
    local log_file=$(_log_file_for_worker "decode" "$i")

    log "Launching decode worker $i on GPUs $gpu_list"
    CUDA_VISIBLE_DEVICES=$gpu_list \
      ${dynamo_args["decode-cmd"]} \
      $(array_to_args decode_args) ${decode_args["--extra-args"]} > $log_file 2>&1 &
  done
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
  [[ "${dynamo_args["decode-nodelist"]}" == *"$name"* ]]
}

_is_prefill_node() {
  local name="$(_current_node_name)"
  [[ "${dynamo_args["prefill-nodelist"]}" == *"$name"* ]]
}

_init_runtime_env() {
  if _is_vllm; then
    export HF_HOME="${HUGGINGFACE_HOME}"
  fi
  export NATS_SERVER="nats://${dynamo_args["frontend-node"]}:${dynamo_args["nats-port"]}"
  export ETCD_ENDPOINTS="http://${dynamo_args["frontend-node"]}:${dynamo_args["etcd-port"]}"
  export UCX_LOG_FILE="${RESULTS_DIR}/ucx_log_%h.log"
  DONE_MARKER="${RESULTS_DIR}/${DONE_MARKER}"
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

function main()
{
  _init_runtime_env

  launch_node_setup_cmd

  if _is_vllm; then
    cd ${dynamo_args["workspace-path"]}
  fi

  if _is_frontend_node; then
    log "Node ID: $SLURM_NODEID, Role: frontend"
    log_node_role "$(_current_node_name)" "frontend"
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
    launch_genai_perf
    touch "$DONE_MARKER"
  fi

  wait_for_frontend_marker
}

parse_args "$@"

log "env: $(env)"

log "Starting main"
main
log "Done with main"
