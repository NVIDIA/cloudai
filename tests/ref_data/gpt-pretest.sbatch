#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH -N 1
#SBATCH --partition=main

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export COMBINE_THRESHOLD=1
export PER_GPU_COMBINE_THRESHOLD=0
export XLA_FLAGS="--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD --xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD --xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD"

srun \
--mpi=pmix \
-N 8 \
-o __OUTPUT_DIR__/output_pretest-%j-%n-%t.txt \
-e __OUTPUT_DIR__/error_pretest-%j-%n-%t.txt \
--container-image=nvcr.io/nvidia/pytorch:24.02-py3 \
/usr/local/bin/all_gather_perf_mpi \
--nthreads 1 \
--ngpus 1 \
--minbytes 8M \
--maxbytes 16G \
--stepbytes 1M \
--op sum \
--datatype float \
--root 0 \
--iters 20 \
--warmup_iters 5 \
--agg_iters 1 \
--average 1 \
--parallel_init 0 \
--check 1 \
--blocking 1 \
--cudagraph 0 \
--stepfactor 2
PRETEST_OUTPUT_FILES="__OUTPUT_DIR__/output_pretest-*.txt"
keyword="Avg bus bandwidth"

# Use grep to search for the keyword in the files
if grep -q "$keyword" $PRETEST_OUTPUT_FILES; then
    PRE_TEST_SUCCESS=true
fi
if [ "$PRE_TEST_SUCCESS" = true ]; then
    echo "Loading container with srun command"
    srun --mpi=none --container-image=https://docker/url --container-name=cont true
    echo "Running srun command"
    srun \
    --mpi=none \
     \
    --export=ALL \
    -o __OUTPUT_DIR__/output-%j-%n-%t.txt \
    -e __OUTPUT_DIR__/error-%j-%n-%t.txt \
    --container-name=cont \
    --container-mounts=__OUTPUT_DIR__:/opt/paxml/workspace/ \
    /opt/paxml/workspace/run.sh
fi