#! /bin/bash

# Called as:
    #   bash ./custom_workload.sh --result_dir <result_dir> --report_file <report_file> --calc_percentile_csv_script <calc_percentile_csv_script> --gpus_per_node <gpus_per_node> -- <lmbench-cmdline-args>

# extract result_dir, report_file, and calc_percentile_csv_script from the command line arguments
result_dir=""
report_name="custom_workload_report.csv"
model=""
url=""
port=""
endpoint=""
connector=""
kvbm_metrics_port=""
all_isl=""
declare -A workload_args


# Simple log function
log() {
  echo "[$(date +%F\ %T) $(hostname)]: $*"
}

function parse_custom_workload_args()
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
    --connector)
      connector="$2"
      shift 2
      ;;
    --kvbm_metrics_port)
      kvbm_metrics_port="$2"
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
    --)
      shift
      parse_custom_workload_args "$@"
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

  log """Parsed args:
    model: $model
    url: $url
    port: $port
    endpoint: $endpoint
    connector: $connector
    kvbm_metrics_port: $kvbm_metrics_port
    result_dir: $result_dir
    install_dir: $install_dir
    report_name: $report_name
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

function clear_kv_cache()
{
  # All ports are passed via command line arguments from ai_dynamo.sh
  # - port: Dynamo frontend API port (for /clear_kv_blocks endpoint)
  # - kvbm_metrics_port: KVBM metrics port (for cache hit rate metrics)

  if [[ -z "$port" ]]; then
    log "ERROR: API port not specified, skipping KV cache clear"
    return 1
  fi

  local api_endpoint="${url}:${port}"

  # Check KV cache status before clearing (metrics on main API port)
  status=$(curl -s ${api_endpoint}/metrics 2>/dev/null | grep -E "kvstats_active_blocks|kvstats_total_blocks" || echo "metrics unavailable")
  log "KV cache status before clear: $status"

  # Clear KV blocks via the dynamo HTTP endpoint
  # This internally calls reset_prefix_cache() on all workers
  response=$(curl -s -X POST ${api_endpoint}/clear_kv_blocks 2>/dev/null || echo "endpoint unavailable")
  log "KV blocks cleared. Response: $response"

  # Check KV cache status after clearing
  status=$(curl -s ${api_endpoint}/metrics 2>/dev/null | grep -E "kvstats_active_blocks|kvstats_total_blocks" || echo "metrics unavailable")
  log "KV cache status after clear: $status"

  # Check KVBM-specific metrics if connector is kvbm and port is specified
  if [[ "$connector" == "kvbm" && -n "$kvbm_metrics_port" ]]; then
    local kvbm_metrics_endpoint="${url}:${kvbm_metrics_port}/metrics"
    status=$(curl -s ${kvbm_metrics_endpoint} 2>/dev/null | grep -E "host_cache_hit_rate|disk_cache_hit_rate" || echo "kvbm metrics unavailable")
    log "KVBM cache hit rates after clear: $status"
  fi

  # TODO: Add LMCache clearing when connector is lmcache
  # if [[ "$connector" == "lmcache" ]]; then
  #   clear_lmcache
  # fi
}

function main()
{
  process_args "$@"

  report_file=$result_dir/$report_name

  log "Launching custom workload with ISLs: $all_isl"

  python3 ${install_dir}/openai_chat_client.py --dump_csv_header --out $report_file

  for isl in $(echo $all_isl | tr ',' '\n'); do
    log "Launching custom workload with ISL: $isl"
    python3 ${install_dir}/openai_chat_client.py \
      --model $model \
      --url $url:$port/v1 \
      --isl $isl \
      --osl 10 \
      --out $report_file \
      --num_filler_prompts 100 \
      --filler_len_chars 10000

    clear_kv_cache
  done

  log "Done with custom workload run"
}

main "$@"
