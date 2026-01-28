#! /bin/bash

# Called as:
    #   bash ./custom_bench.sh --result_dir <result_dir> --report_file <report_file> --calc_percentile_csv_script <calc_percentile_csv_script> --gpus_per_node <gpus_per_node> -- <lmbench-cmdline-args>

# extract result_dir, report_file, and calc_percentile_csv_script from the command line arguments
result_dir=""
report_name="custom_bench_report.csv"
model=""
url=""
port=""
endpoint=""
cmdline_args=()


# Simple log function
log() {
  echo "[$(date +%F\ %T) $(hostname)]: $*"
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
    --result_dir)
      result_dir="$2"
      shift 2
      ;;
    --report_name)
      report_name="$2"
      shift 2
      ;;
    --)
      shift
      cmdline_args=("$@")
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
  local dyn_metrics_endpoint="${url}:${DYN_METRICS_PORT}/metrics"
  local kvbm_metrics_endpoint="${url}:${DYN_KVBM_METRICS_PORT}/metrics"

  # This clears G1 (GPU) + G2 (CPU) + G3 (Disk) at once
  status=$(curl -s ${dyn_metrics_endpoint} | grep -E "kvstats_active_blocks|kvstats_total_blocks")
  log "KV cache status before clear: $status"

  response=$(curl -s -X POST ${dyn_metrics_endpoint}/reset_prefix_cache)
  log "KV prefix cache reset. Response: $response"

  response=$(curl -s -X POST ${dyn_metrics_endpoint}/clear_kv_blocks)
  log "KV blocks cleared. Response: $response"

  status=$(curl -s ${dyn_metrics_endpoint} | grep -E "kvstats_active_blocks|kvstats_total_blocks")
  log "KV cache status after clear: $status"

  status=$(curl -s ${kvbm_metrics_endpoint} | grep -E "host_cache_hit_rate|disk_cache_hit_rate")
  log "KVBM cache hit rates after clear: $status"

  # if [[ "${dynamo_args["connector"]}" == "lmcache" ]]; then
  #   clear_lmcache
  # fi
}

function main()
{
  process_args "$@"

  report_file=$result_dir/$report_name

  all_isl="${cmdline_args["isl"]}"
  log "Launching custom bench with ISLs: $all_isl"

  # Split comma-separated all_isl into individual isl values and iterate over them. Use cut -d',' to split.
  echo "isl,context_tokens,baseline_cached_tokens,baseline_ttft_seconds,no_flush_cached_tokens,no_flush_ttft_seconds,post_flush_cached_tokens,post_flush_ttft_seconds" >> $report_file

  for isl in $(echo $all_isl | tr ',' '\n'); do
    log "Launching custom bench with ISL: $isl"
    python3 /cloudai_install/openai_chat_client.py \
      --model $model \
      --url $url:$port/$endpoint \
      --isl $isl \
      --osl 10 \
      --out $report_file \
      --num_filler_prompts 100 \
      --filler_len_chars 10000

    clear_kv_cache
  done

  log "Done with custom bench run"
}

main "$@"