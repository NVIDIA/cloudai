#! /bin/bash

# Called as:
    #   bash ./lmbench.sh --result_dir <result_dir> --report_file <report_file> --calc_percentile_csv_script <calc_percentile_csv_script> --gpus_per_node <gpus_per_node> -- <lmbench-cmdline-args>

# Simple log function
log() {
  echo "[$(date +%F\ %T) $(hostname)]: $*"
}

# extract result_dir, report_file, and calc_percentile_csv_script from the command line arguments
result_dir=""
report_name="lmbench_report.csv"
calc_percentile_csv_script=""
gpus_per_node=1
lmbench_dir="/git/LMBenchmark"
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
    --report_name)
      report_name="$2"
      shift 2
      ;;
    --extra_args)
      extra_args="$2"
      shift 2
      ;;
    --repo)
      lmbench_dir="$2"
      shift 2
      ;;
    --calc_percentile_csv_script)
      calc_percentile_csv_script="$2"
      shift 2
      ;;
    --gpus_per_node)
      gpus_per_node="$2"
      shift 2
      ;;
    --cmd)
      cmd="$2"
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

pushd $lmbench_dir

# launch lmbench
log "Launching lmbench with args: $cmdline_args"


# run LMBenchmark, adjust the model name if you are using a different model
# for detail how to config and run LMBenchmark: https://github.com/LMCache/LMBenchmark/tree/main/synthetic-multi-round-qa

#export NUM_USERS_WARMUP="20"
#export NUM_USERS="15"
#export NUM_ROUNDS="20"
#export SYSTEM_PROMPT="1000" # Shared system prompt length
#export CHAT_HISTORY="7000" # User specific chat history length
#export ANSWER_LEN="100" # Generation length per round
#export INIT_USER_ID="1"
#export TEST_DURATION="600" # Duration of the test in seconds

log "Launching lmbench with args: $cmd $cmdline_args $extra_args"

eval "$cmd $cmdline_args $extra_args"

log "Done with lmbench run; summarizing results"

for i in $RESULTS_DIR/${key}_*.csv; do
    python3 ${calc_percentile_csv_script} $i -o $RESULTS_DIR/lmbench_${key}_report.csv
done

log "Done with lmbench run"
popd
