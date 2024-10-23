#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH -N 1
#SBATCH --partition=main

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export COMBINE_THRESHOLD=1
export PER_GPU_COMBINE_THRESHOLD=0
export XLA_FLAGS="--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD --xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD --xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD"

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
