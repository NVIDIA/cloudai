#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH -N 1
#SBATCH --partition=main

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export COMBINE_THRESHOLD=1
export PER_GPU_COMBINE_THRESHOLD=0
export XLA_FLAGS="--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD --xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD --xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD"

srun --output=__OUTPUT_DIR__/output/pre_test/nccl/stdout.txt --error=__OUTPUT_DIR__/output/pre_test/nccl/stderr.txt --mpi=pmix --container-image=nvcr.io/nvidia/pytorch:24.02-py3 --container-mounts=__OUTPUT_DIR__/output/pre_test/nccl:/cloudai_run_results /usr/local/bin/all_reduce_perf_mpi --nthreads 1 --ngpus 1 --minbytes 32M --maxbytes 32M --stepbytes 1M --op sum --datatype float --root 0 --iters 20 --warmup_iters 5 --agg_iters 1 --average 1 --parallel_init 0 --check 1 --blocking 0 --cudagraph 0
SUCCESS_0=$(grep -q "Avg bus bandwidth" __OUTPUT_DIR__/output/pre_test/nccl/stdout.txt && echo 1 || echo 0)
PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && echo 1 || echo 0 )
if [ $PRE_TEST_SUCCESS -eq 1 ]; then
        echo "Loading container with srun command"
    srun --mpi=none --container-image=https://docker/url --container-name=cont true
    echo "Running srun command"
    srun \
    --mpi=none \
     \
    --export=ALL \
    -o __OUTPUT_DIR__/output/output-%j-%n-%t.txt \
    -e __OUTPUT_DIR__/output/error-%j-%n-%t.txt \
    --container-name=cont \
    --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/output:/opt/paxml/workspace/ \
    /opt/paxml/workspace/run.sh
fi
