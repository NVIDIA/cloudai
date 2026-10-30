#!/usr/bin/env bash
set -euo pipefail

# Configure submission environment
export TEST_VAR='value with spaces'
export SRTSLURM_CONFIG=__CLOUDAI_DIR__/conf/experimental/srt_slurm/srtslurm.yaml
export CLOUDAI_INSTALL_FOLDER=__INSTALL_DIR__

# Prevent login-node Python environments from leaking into compute jobs.
# A: uv searches PATH after VIRTUAL_ENV. Remove every virtualenv-backed
# PATH entry so it cannot select an incompatible login-node python/python3.
IFS=: read -r -a SRT_PATH_PARTS <<< "$PATH"
SRT_CLEAN_PATH=()
for SRT_PATH_PART in "${SRT_PATH_PARTS[@]}"; do
    if [[ ! -f "$SRT_PATH_PART/../pyvenv.cfg" ]]; then
        SRT_CLEAN_PATH+=("$SRT_PATH_PART")
    fi
done
export PATH="$(IFS=:; printf '%s' "${SRT_CLEAN_PATH[*]}")"
# B: uv prioritizes the active environment. Unset VIRTUAL_ENV so it cannot
# inspect or execute CloudAI's login-node interpreter on a compute node.
unset VIRTUAL_ENV

# Submit the recipe
mkdir -p __OUTPUT_DIR__/output/srt_slurm_outputs
# Clear the old result lookup ID in case this wrapper is manually rerun and submission fails.
rm -f __OUTPUT_DIR__/output/srt_job_id.txt
# C: Slurm inherits this working directory. Leave CloudAI's checkout so
# compute-side uv does not discover its login-node .venv, even with --no-project.
cd __INSTALL_DIR__/srt-slurm__v2.2.1
if ! __INSTALL_DIR__/srt-slurm__v2.2.1-venv/bin/srtctl apply -f __CLOUDAI_DIR__/conf/experimental/srt_slurm/recipe.yaml -o __OUTPUT_DIR__/output/srt_slurm_outputs -y --set 'benchmark.concurrencies="4"' --no-preflight > __OUTPUT_DIR__/output/run_srtctl.log 2>&1; then
    cat __OUTPUT_DIR__/output/run_srtctl.log >&2
    exit 1
fi
cat __OUTPUT_DIR__/output/run_srtctl.log >&2

# Record the submitted job ID for CloudAI
if ! SRT_JOB_ID="$(sed -nE 's/.*Job ([0-9]+) submitted.*/\1/p; s/.*Submitted batch job ([0-9]+).*/\1/p' __OUTPUT_DIR__/output/run_srtctl.log | tail -1)" || [[ ! "$SRT_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Unable to find submitted srt-slurm job ID" >&2
    exit 1
fi
# Persist the result lookup ID now; slurm-job.toml is written at completion
# only when accounting is available. Stdout separately supplies the runner ID.
printf "%s\n" "$SRT_JOB_ID" > __OUTPUT_DIR__/output/srt_job_id.txt
echo "Submitted batch job $SRT_JOB_ID"
