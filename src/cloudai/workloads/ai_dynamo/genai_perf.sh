#! /bin/bash

# Called as:
    #   ./genai_perf_wrapper.sh --result_dir <result_dir> --report_file <report_file> --calc_percentile_csv_script <calc_percentile_csv_script> --gpus_per_node <gpus_per_node> -- <genai-perf-cmdline-args>

# Simple log function
log() {
  echo "[$(date +%F\ %T) $(hostname)]: $*"
}

# extract result_dir, report_file, and calc_percentile_csv_script from the command line arguments
result_dir=""
report_name="genai_perf_report.csv"
calc_percentile_csv_script=""
gpus_per_node=1
port=""
repo=""
cmd=""
cmdline_args=()
extra_args=""

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
    --extra-args|--extra_args)
      extra_args="$2"
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

# Combine cmdline_args and extra_args
cmdline_args="${cmdline_args[*]} $extra_args"

# Build the full command with model and url
full_cmd="$cmd $cmdline_args"

# launch genai-perf
log "Launching genai-perf with args: $full_cmd"

eval $full_cmd

log "Done with genai-perf run"

# Calculate total GPUs - use SLURM_JOB_NUM_NODES if available, otherwise default to 1 node
num_nodes=${SLURM_JOB_NUM_NODES:-1}
total_gpus=$(( $gpus_per_node * $num_nodes ))

profile_path=$(find $result_dir -type f -name "profile_genai_perf.csv" -print -quit)
sed -ie 's/\r//g' $profile_path
if [[ -f "$profile_path" ]]; then
    output_tokens_per_second=$(grep "Output Token Throughput (tokens/sec)" $profile_path | cut -d ',' -f 2)
    output_tokens_per_second_per_gpu=$(awk "BEGIN {printf \"%.2f\", $output_tokens_per_second / $total_gpus}")
    request_throughput=$(grep "Request Throughput (per sec)" $profile_path | cut -d ',' -f 2)
    request_count=$(grep "Request Count (count)" $profile_path | cut -d ',' -f 2)
    grep ".*,.*,.*,.*" $profile_path > $result_dir/$report_name
    echo "Output tokens per second per gpu,$output_tokens_per_second_per_gpu,0,0,0,0,0,0,0,0,0,0,0" >> $result_dir/$report_name
    echo "Request throughput per second,$request_throughput,0,0,0,0,0,0,0,0,0,0,0" >> $result_dir/$report_name
    echo "Request count,$request_count,0,0,0,0,0,0,0,0,0,0,0" >> $result_dir/$report_name
fi
