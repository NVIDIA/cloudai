#! /bin/bash

# Called as:
    #   bash ./kvstorage.sh --result_dir <result_dir> --report_name <report_name> -- <kvstorage-cmdline-args>

# extract result_dir, report_file, and calc_percentile_csv_script from the command line arguments
result_dir=""
report_name="kvstorage_report.csv"
model=""
url=""
port=""
endpoint=""
connector=""
kvbm_metrics_port=""
all_isl=""
declare -A workload_args
kv_cache_token_size=0
num_filler_tokens=32000
g1_token_size=0
g2_token_size=0
g3_token_size=0
bytes_per_token=0
dyn_system_port="${DYN_SYSTEM_PORT:-8081}"
client_script="./kvstorage.py"


# Simple log function
log() {
  echo "[$(date +%F\ %T) $(hostname)]: $*"
}

function parse_kvstorage_args()
{
  local args="$@"
  while [[ $# -gt 0 ]]; do
    case "$1" in
    --*)
      workload_args["${1}"]="$2"
      shift 2
      ;;
    *)
      shift
      ;;
    esac
  done
}

function process_args()
{
  while [[ $# -gt 0 ]]; do
    case "$1" in
    --model)
      model="$2"
      shift 2
      ;;
    --url)
      url="$2"
      shift 2
      ;;
    --port)
      port="$2"
      shift 2
      ;;
    --endpoint)
      endpoint="$2"
      shift 2
      ;;
    --decode-connector)
      decode_connector="$2"
      shift 2
      ;;
    --prefill-connector)
      prefill_connector="$2"
      shift 2
      ;;
    --kvbm_metrics_port)
      kvbm_metrics_port="$2"
      shift 2
      ;;
    --dyn_system_port)
      dyn_system_port="$2"
      shift 2
      ;;
    --result_dir)
      result_dir="$2"
      shift 2
      ;;
    --install_dir)
      install_dir="$2"
      shift 2
      ;;
    --report_name)
      report_name="$2"
      shift 2
      ;;
    --isl)
      all_isl="$2"
      shift 2
      ;;
    --kv_cache_token_size)
      kv_cache_token_size="$2"
      shift 2
      ;;
    --num_filler_tokens)
      num_filler_tokens="$2"
      shift 2
      ;;
    --)
      shift
      parse_kvstorage_args "$@"
      break
      ;;
    --*)
      shift 2
      ;;
    *)
      shift
      ;;
    esac
  done

  client_script="${install_dir}/kvstorage.py"

  log """Parsed args:
    model: $model
    url: $url
    port: $port
    endpoint: $endpoint
    decode-connector: $decode_connector
    prefill-connector: $prefill_connector
    kvbm_metrics_port: $kvbm_metrics_port
    result_dir: $result_dir
    install_dir: $install_dir
    report_name: $report_name
    kv_cache_token_size: $kv_cache_token_size
    num_filler_tokens: $num_filler_tokens
    isl: $all_isl
    workload_args: $(for key in "${!workload_args[@]}"; do echo -n "$key: ${workload_args[$key]} "; done)
  """
}

#function clear_lmcache()
#{
#  log "Clearing LMCache"
#
#  response=$(curl -X POST http://${lmcache_config["controller_url"]}/clear \
#  -H "Content-Type: application/json" \
#  -d '{
#    "instance_id": "lmcache_default_instance",
#    "location": "LocalCPUBackend"
#  }')
#
#  log "LMCache cleared. Response: $response"
#}

function print_metrics()
{
  local frontend_metrics_endpoint="${url}:${port}/metrics"
  local component_metrics_endpoint="${url}:${dyn_system_port}/metrics"
  local kvbm_metrics_endpoint="${url}:${kvbm_metrics_port}/metrics"

  #status=$(curl -s ${frontend_metrics_endpoint} 2>/dev/null | grep -E "cache.*model=") # | grep -E "kvstats_active_blocks|kvstats_total_blocks" || echo "metrics unavailable")
  #log "Frontend metrics: $status"

  #status=$(curl -s ${component_metrics_endpoint} 2>/dev/null | grep -E "cache.*model=") # | grep -E "kvstats_active_blocks|kvstats_total_blocks" || echo "metrics unavailable")
  #log "Component metrics: $status"

  status=$(curl -s ${kvbm_metrics_endpoint} 2>/dev/null ) # | grep -E "host_cache_hit_rate|disk_cache_hit_rate" || echo "kvbm metrics unavailable")
  log "KVBM metrics: $status"
}

function clear_kv_cache()
{
  if [[ -z "$port" ]]; then
    log "ERROR: API port not specified, skipping KV cache clear"
    return 1
  fi

  log "Metrics before clear:"
  print_metrics

  # Clear KV blocks via the dynamo HTTP endpoint
  # This internally calls reset_prefix_cache() on all workers
  response=$(curl -s -X POST ${url}:${dyn_system_port}/engine/clear_kv_blocks 2>/dev/null || echo "endpoint unavailable")
  log "KV blocks cleared. Response: $response"

  log "Metrics after clear:"
  print_metrics

  # TODO: Add LMCache clearing when connector is lmcache
  # if [[ "$connector" == "lmcache" ]]; then
  #   clear_lmcache
  # fi
}

function compute_kv_cache_token_size_from_log()
{
  # Parse the decode worker log to extract G1 (GPU) KV cache information
  # Log format examples:
  #   INFO gpu_worker.determine_available_memory: Available KV cache memory: 54.90 GiB  (per GPU!)
  #   INFO kv_cache_utils._report_kv_cache_config: GPU KV cache size: 3,198,272 tokens  (total for worker)
  #
  # IMPORTANT: "Available KV cache memory" is PER GPU, while "GPU KV cache size" is for the entire
  # worker (total tokens across all GPUs). We need to multiply memory by tensor_parallel_size.

  log "Computing KV cache token sizes from worker log files..."

  # Find decode worker log file(s) in result_dir
  local decode_log=$(find "$result_dir" -name "dynamo_decode_0_0.log" 2>/dev/null | head -1)
  if [[ -z "$decode_log" ]]; then
    log "WARNING: No decode worker log found in $result_dir, falling back to query-based method"
    return 1
  fi

  log "Using decode worker log: $decode_log"

  # Extract tensor_parallel_size from log: "tensor_parallel_size=8"
  local tp_size=1
  local tp_line=$(grep -o "tensor_parallel_size=[0-9]*" "$decode_log" | head -1)
  if [[ -n "$tp_line" ]]; then
    tp_size=$(echo "$tp_line" | cut -d'=' -f2)
    log "Tensor parallel size from log: $tp_size GPUs"
  else
    log "WARNING: Could not find tensor_parallel_size in log, assuming 1 GPU"
  fi

  # Extract G1 token count: "GPU KV cache size: 3,198,272 tokens"
  # This is the TOTAL token capacity for the entire worker (across all GPUs)
  local g1_tokens_line=$(grep "GPU KV cache size:" "$decode_log" | tail -1)
  if [[ -z "$g1_tokens_line" ]]; then
    log "WARNING: Could not find 'GPU KV cache size' in log, falling back to query-based method"
    return 1
  fi

  # Parse: extract number, remove commas
  g1_token_size=$(echo "$g1_tokens_line" | sed -E 's/.*GPU KV cache size: ([0-9,]+) tokens.*/\1/' | tr -d ',')
  log "G1 (GPU) token size from log: $g1_token_size tokens (total for worker)"

  # Extract G1 memory size per GPU: "Available KV cache memory: 54.90 GiB"
  # This is PER GPU, so we need to multiply by tensor_parallel_size
  local g1_memory_line=$(grep "Available KV cache memory:" "$decode_log" | tail -1)
  if [[ -z "$g1_memory_line" ]]; then
    log "WARNING: Could not find 'Available KV cache memory' in log, falling back to query-based method"
    return 1
  fi

  # Parse: extract the GiB value (this is per GPU)
  local g1_memory_per_gpu_gib=$(echo "$g1_memory_line" | sed -E 's/.*Available KV cache memory: ([0-9.]+) GiB.*/\1/')
  log "G1 (GPU) memory per GPU from log: $g1_memory_per_gpu_gib GiB"

  # Calculate total G1 memory across all GPUs
  local g1_memory_total_gib=$(awk "BEGIN {printf \"%.2f\", $g1_memory_per_gpu_gib * $tp_size}")
  log "G1 (GPU) total memory across $tp_size GPUs: $g1_memory_total_gib GiB"

  # Calculate bytes per token = (total_G1_GiB * 1024^3) / G1_tokens
  # Using awk to handle the initial float-to-int conversion from g1_memory_per_gpu_gib
  bytes_per_token=$(awk "BEGIN {printf \"%d\", ($g1_memory_total_gib * 1024 * 1024 * 1024) / $g1_token_size}")
  log "Calculated bytes per token: $bytes_per_token"

  # Calculate G2 (CPU) token size from DYN_KVBM_CPU_CACHE_GB environment variable
  local g2_cache_gb=${DYN_KVBM_CPU_CACHE_GB:-0}
  if [[ "$g2_cache_gb" != "0" && -n "$g2_cache_gb" ]]; then
    # G2_tokens = (G2_GB * 1024^3) / bytes_per_token
    g2_token_size=$(( (g2_cache_gb * 1024 * 1024 * 1024) / bytes_per_token ))
    log "G2 (CPU) cache: $g2_cache_gb GB = $g2_token_size tokens"
  else
    log "G2 (CPU) cache not configured (DYN_KVBM_CPU_CACHE_GB not set)"
  fi

  # Calculate G3 (Disk) token size from DYN_KVBM_DISK_CACHE_GB environment variable
  local g3_cache_gb=${DYN_KVBM_DISK_CACHE_GB:-0}
  if [[ "$g3_cache_gb" != "0" && -n "$g3_cache_gb" ]]; then
    # G3_tokens = (G3_GB * 1024^3) / bytes_per_token
    g3_token_size=$(( (g3_cache_gb * 1024 * 1024 * 1024) / bytes_per_token ))
    log "G3 (Disk) cache: $g3_cache_gb GB = $g3_token_size tokens"
  else
    log "G3 (Disk) cache not configured (DYN_KVBM_DISK_CACHE_GB not set)"
  fi

  kv_cache_token_size=$(( g1_token_size + g2_token_size ))

  log "KV cache summary:"
  log "  G1 (GPU):  $g1_token_size tokens (${g1_memory_per_gpu_gib} GiB/GPU x $tp_size GPUs = ${g1_memory_total_gib} GiB total)"
  log "  G2 (CPU):  $g2_token_size tokens (${g2_cache_gb} GB)"
  log "  G3 (Disk): $g3_token_size tokens (${g3_cache_gb} GB)"
  log "  Total:     $kv_cache_token_size tokens"
  log "  Bytes per token: $bytes_per_token"
  return 0
}

function compute_kv_cache_token_size_from_query()
{
  # Fallback: compute by sending queries (original method)
  local kv_cache_token_size_file=$result_dir/kv_cache_token_size.out
  log "Computing KV cache token size via queries..."
  python3 $client_script \
      --model $model \
      --url $url:$port/v1 \
      --osl 10 \
      --out $kv_cache_token_size_file \
      --compute_kv_cache_token_size \
      --num_filler_tokens $num_filler_tokens \
      --max_filler_prompts 200 \
      --min_filler_prompts 10

  kv_cache_token_size=$(grep cache $kv_cache_token_size_file | cut -d':' -f 2 | tr -d ' ')
  log "KV cache token size from queries: $kv_cache_token_size"
}

function compute_kv_cache_token_size()
{
  if [[ $kv_cache_token_size -gt 0 ]]; then
    log "KV cache token size already provided: $kv_cache_token_size"
    return
  fi

  # Try to get from log files first (faster, no queries needed)
  if compute_kv_cache_token_size_from_log; then
    log "Successfully computed KV cache token size from log files"
  else
    # Fallback to query-based method
    log "Falling back to query-based KV cache token size computation"
    compute_kv_cache_token_size_from_query
  fi
}

function main()
{
  process_args "$@"

  report_file=$result_dir/$report_name

  compute_kv_cache_token_size
  local num_filler_prompts=$(( 1 + (kv_cache_token_size / num_filler_tokens) ))

  log "Dumping CSV header"
  python3 $client_script --dump_csv_header --out $report_file

  log "Launching KV storage workload with ISLs: $all_isl"
  for isl in $(echo $all_isl | tr ',' '\n'); do
    log "Launching KV storage workload with ISL: $isl"
    python3 $client_script \
      --model $model \
      --url $url:$port/v1 \
      --isl $isl \
      --osl 1 \
      --out $report_file \
      --num_filler_prompts $num_filler_prompts \
      --num_filler_tokens $num_filler_tokens

    log "Sleeping for 5 seconds before clearing KV cache"
    sleep 5
    clear_kv_cache
  done

  log "Done with KV storage workload run"
}

main "$@"
