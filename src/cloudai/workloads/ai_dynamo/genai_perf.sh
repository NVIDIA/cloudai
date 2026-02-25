#! /bin/bash

# Called as:
    #   ./genai_perf.sh --result_dir <result_dir> --report_file <report_file> --gpus_per_node <gpus_per_node> -- <genai-perf-cmdline-args>

result_dir=""
report_name="genai_perf_report.csv"
gpus_per_node=1
cmd=""
extra_args=""
declare -A genai_perf_args

# Simple log function
log() {
  echo "[$(date +%F\ %T) $(hostname)]: $*"
}

function parse_genai_perf_args()
{
  while [[ $# -gt 0 ]]; do
    case "$1" in
    --*)
      genai_perf_args["${1}"]="$2"
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
  local url="http://localhost"
  local port=8000
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        genai_perf_args["--model"]="$2"
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
        genai_perf_args["--endpoint"]="$2"
        shift 2
        ;;
      --result-dir)
        result_dir="$2"
        shift 2
        ;;
      --gpus-per-node)
        gpus_per_node="$2"
        shift 2
        ;;
      --report-name)
        report_name="$2"
        shift 2
        ;;
      --cmd)
        cmd="$2"
        shift 2
        ;;
      --extra-args)
        extra_args="$2"
        shift 2
        ;;
      --)
        shift
        parse_genai_perf_args "$@"
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

  genai_perf_args["--url"]="$url:$port"
  genai_perf_args["--artifact-dir"]="$result_dir/genai_perf_artifacts"
  genai_perf_args["--profile-export-file"]="profile.json"

  log "Parsed args:
    result_dir: $result_dir
    report_name: $report_name
    cmd: $cmd
    extra_args: $extra_args
    genai_perf_args: $(for key in "${!genai_perf_args[@]}"; do echo "        $key ${genai_perf_args[$key]} "; done)"
}

function process_results()
{
  # Calculate total GPUs - use SLURM_JOB_NUM_NODES if available, otherwise default to 1 node
  local num_nodes=${SLURM_JOB_NUM_NODES:-1}
  local total_gpus=$(( $gpus_per_node * $num_nodes ))

  log "Processing results from $result_dir"
  local profile_path=$(find "$result_dir" -type f -name "profile_genai_perf.csv" -print -quit)
  if [[ -f "$profile_path" ]]; then
    log "Processing results from $profile_path"
    sed -i 's/\r//g' "$profile_path"
    local output_tokens_per_second=$(grep "Output Token Throughput (tokens/sec)" "$profile_path" | cut -d ',' -f 2)
    local output_tokens_per_second_per_gpu=$(awk "BEGIN {printf \"%.2f\", $output_tokens_per_second / $total_gpus}")
    local request_throughput=$(grep "Request Throughput (per sec)" "$profile_path" | cut -d ',' -f 2)
    local request_count=$(grep "Request Count (count)" "$profile_path" | cut -d ',' -f 2)
    grep ".*,.*,.*,.*" "$profile_path" > "$result_dir/$report_name"
    echo "Output tokens per second per gpu,$output_tokens_per_second_per_gpu,0,0,0,0,0,0,0,0,0,0,0" >> "$result_dir/$report_name"
    echo "Request throughput per second,$request_throughput,0,0,0,0,0,0,0,0,0,0,0" >> "$result_dir/$report_name"
    echo "Request count,$request_count,0,0,0,0,0,0,0,0,0,0,0" >> "$result_dir/$report_name"
    log "Successfully generated report at $result_dir/$report_name"
  else
    log "Failed to generate report: No profile_genai_perf.csv found"
    exit 1
  fi
}

function main()
{
  process_args "$@"

  # Combine genai_perf_args (key-value pairs) and extra_args
  cmdline_args=""
  for key in "${!genai_perf_args[@]}"; do
    local val="${genai_perf_args[$key]}"
    # Quote values that contain spaces so eval doesn't split them
    if [[ "$val" == *" "* ]]; then
      val="${val//\"/\\\"}"  # Escape existing quotes
      cmdline_args+="$key \"${val}\" "
    else
      cmdline_args+="$key ${val} "
    fi
  done
  cmdline_args+="$extra_args"

  # Build the full command with model and url
  full_cmd="$cmd $cmdline_args"

  # launch genai-perf
  log "Launching genai-perf with args: $full_cmd"

  eval "$full_cmd"

  log "Done with genai-perf run"

  process_results
}

main "$@"
exit 0
