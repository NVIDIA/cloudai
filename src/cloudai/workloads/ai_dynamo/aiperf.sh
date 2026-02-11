#! /bin/bash

# Called as:
    #   ./aiperf.sh --result_dir <result_dir> --report_file <report_file> --calc_percentile_csv_script <calc_percentile_csv_script> --gpus_per_node <gpus_per_node> -- <aiperf-cmdline-args>

# extract result_dir, report_file, and calc_percentile_csv_script from the command line arguments
result_dir=""
report_name="aiperf_report.csv"
gpus_per_node=1
port=""
cmd=""
extra_args=""
declare -A aiperf_args
decode_nodes=""
aiperf_profile_csv="profile_export_aiperf.csv"
metrics_urls=""
version=""

# Simple log function
log() {
  echo "[$(date +%F\ %T) $(hostname)]: $*"
}

function parse_aiperf_args()
{
  while [[ $# -gt 0 ]]; do
    case "$1" in
    --*)
      aiperf_args["${1}"]="$2"
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
      --result_dir)
        result_dir="$2"
        shift 2
        ;;
      --install_dir)
        install_dir="$2"
        shift 2
        ;;
      --gpus_per_node)
        gpus_per_node="$2"
        shift 2
        ;;
      --report_name)
        report_name="$2"
        shift 2
        ;;
      --cmd)
        cmd="$2"
        shift 2
        ;;
      --version)
        version="$2"
        shift 2
        ;;
      --decode-nodes)
        decode_nodes="$2"
        shift 2
        ;;
      --extra-args|--extra_args)
        extra_args="$2"
        shift 2
        ;;
      --)
        shift
        parse_aiperf_args "$@"
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
    result_dir: $result_dir
    install_dir: $install_dir
    report_name: $report_name
    cmd: $cmd
    version: $version
    extra_args: $extra_args
    decode_nodes: $decode_nodes
    aiperf_args:
        $(for key in "${!aiperf_args[@]}"; do echo "$key ${aiperf_args[$key]} "; done)
  """
}

function update_aiperf_version()
{
  if [[ -n "$version" ]]; then
    log "Updating aiperf version from $(aiperf --version) to $version"
    pip install --upgrade $version
    log "Updated aiperf version to $(aiperf --version)"
  fi
}

function _resolve_server_metrics_auto()
{
  # Auto-discover Prometheus metrics endpoints for Dynamo deployments
  # Returns space-separated list of URLs for --server-metrics
  
  # Frontend metrics (port from dynamo config)
  local frontend_url="http://${url}:${port}/metrics"
  metrics_urls="$frontend_url"
  
  local IFS_SAVE="$IFS"
  IFS=','
  for node in $decode_nodes; do
    local decode_url="http://${node}:9090/metrics"
    metrics_urls="$metrics_urls $decode_url"
  done
  IFS="$IFS_SAVE"
  
  log "Auto-discovered server-metrics URLs: $metrics_urls"
}

function process_result()
{
  local profile_path
  profile_path=$(find "$result_dir" -type f -name "$aiperf_profile_csv" -print -quit)
  if [[ ! -f "$profile_path" ]]; then
    log "WARNING: aiperf profile CSV not found: $aiperf_profile_csv"
    return
  fi

  local num_sections=1
  local has_content=0
  local output_prefix="${result_dir}/aiperf_section"

  while IFS= read -r line; do
    # Strip carriage returns
    line="${line//$'\r'/}"
    if [[ -z "$line" ]]; then
      # Only advance section if the current one had content
      if [[ "$has_content" -eq 1 ]]; then
        num_sections=$(( num_sections + 1 ))
        has_content=0
      fi
    else
      echo "$line" >> "${output_prefix}.${num_sections}.csv"
      has_content=1
    fi
  done < "$profile_path"

  log "Split aiperf CSV into $num_sections section(s)"

  # Section 1: per-request percentile metrics → main report
  if [[ -f "${output_prefix}.1.csv" ]]; then
    mv "${output_prefix}.1.csv" "$report_file"
  fi

  # Section 2: summary metrics
  if [[ -f "${output_prefix}.2.csv" ]]; then
    mv "${output_prefix}.2.csv" "${result_dir}/aiperf_summary.csv"
  fi

  # Section 3: server/GPU metrics
  if [[ -f "${output_prefix}.3.csv" ]]; then
    mv "${output_prefix}.3.csv" "${result_dir}/aiperf_server_metrics.csv"
  fi
}

function main()
{
  process_args "$@"

  report_file=$result_dir/$report_name

  update_aiperf_version

  # Handle server-metrics = "auto" - auto-discover endpoints
  if [[ "${aiperf_args["--server-metrics"]}" == "auto" ]]; then
    _resolve_server_metrics_auto
    aiperf_args["--server-metrics"]="$metrics_urls"
  fi
  
  # Combine aiperf_args (key-value pairs) and extra_args
  cmdline_args=""
  for key in "${!aiperf_args[@]}"; do
    local val="${aiperf_args[$key]}"
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
  
  # launch aiperf
  log "Launching aiperf with args: $full_cmd"
  
  eval "$full_cmd"
  
  log "Done with aiperf run"
  
  process_result
}

main "$@"