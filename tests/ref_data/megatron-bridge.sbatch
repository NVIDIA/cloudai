#!/usr/bin/env bash
set -o pipefail

export NEMORUN_HOME="__OUTPUT_DIR__/output"
mkdir -p "$NEMORUN_HOME"
LOG="__OUTPUT_DIR__/output/cloudai_megatron_bridge_launcher.log"
WRAPPER_STDOUT="__OUTPUT_DIR__/output/cloudai_megatron_bridge_wrapper.stdout"
WRAPPER_STDERR="__OUTPUT_DIR__/output/cloudai_megatron_bridge_wrapper.stderr"
exec > >(tee -a "$WRAPPER_STDOUT") 2> >(tee -a "$WRAPPER_STDERR" >&2)


: >"$LOG"
LAUNCH_RC=0
NEMORUN_HOME="__OUTPUT_DIR__/output" __INSTALL_DIR__/Run__main-venv/bin/python __INSTALL_DIR__/Megatron-Bridge__main/scripts/performance/setup_experiment.py -p main -i __OUTPUT_DIR__/output/megatron_bridge_image.sqsh -hf dummy_token -ng 8 -gn 4 -cb 'export CUDA_VISIBLE_DEVICES=0,1,2,3' -cb 'export NCCL_DEBUG=INFO' -m qwen3 -mr 30b_a3b >>"$LOG" 2>&1 || LAUNCH_RC=$?


JOB_ID=""
JOB_ID=$(grep -Eio "(Job id[: ]+[0-9]+|-[ ]*Job id[: ]+[0-9]+)" "$LOG" | tail -n1 | grep -Eo "[0-9]+" | tail -n1 || true)


if [ -n "${JOB_ID}" ]; then
  if [ "${LAUNCH_RC}" -ne 0 ]; then
    echo "Megatron-Bridge launcher exited non-zero (${LAUNCH_RC}) after submitting job ${JOB_ID}." >&2
    tail -n 20 "$LOG" >&2 || true
    exit "${LAUNCH_RC}"
  fi
  echo "Submitted batch job ${JOB_ID}"
else
  echo "Failed to retrieve job ID." >&2
  if [ "${LAUNCH_RC}" -ne 0 ]; then
    echo "Launcher exit code: ${LAUNCH_RC}" >&2
  fi
  tail -n 20 "$LOG" >&2 || true
  exit 1
fi
