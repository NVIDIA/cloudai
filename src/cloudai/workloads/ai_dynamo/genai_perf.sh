#! /bin/bash

# Called as:
    #   ./genai_perf_wrapper.sh --result_dir <result_dir> --report_file <report_file> --calc_percentile_csv_script <calc_percentile_csv_script> --gpus_per_node <gpus_per_node> -- <genai-perf-cmdline-args>

# Simple log function
log() {
  echo "[$(date +%F\ %T) $(hostname)]: $*"
}

# extract result_dir, report_file, and calc_percentile_csv_script from the command line arguments
result_dir=""
report_file="genai_perf_report.csv"
calc_percentile_csv_script=""
gpus_per_node=1
repo=""
cmdline_args=()

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

cmdline_args="${cmdline_args[*]}"

# launch genai-perf
log "Launching genai-perf with args: $cmdline_args"

${cmdline_args}

log "Done with genai-perf run"

# Calculate total GPUs - use SLURM_JOB_NUM_NODES if available, otherwise default to 1 node
num_nodes=${SLURM_JOB_NUM_NODES:-1}
total_gpus=$(( $gpus_per_node * $num_nodes ))

profile_path=$(find $result_dir -type f -name "profile_genai_perf.csv" -print -quit)
if [[ -f "$profile_path" ]]; then
    python3 ${install_dir}/calc_percentile_csv.py $profile_path -o $result_dir/$report_file
    output_tokens_per_second=$(grep "output_tokens_per_second" $profile_path | awk '{print $2}')
    output_tokens_per_second_per_gpu=$(( $output_tokens_per_second / $total_gpus ))
    grep ".*,.*,.*,.*" $profile_path > $result_dir/$report_file
    echo "Output tokens per second per gpu,$output_tokens_per_second_per_gpu,0,0,0,0,0,0,0,0,0,0,0" >> $result_dir/$report_file
fi

