#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH -N 1
#SBATCH --output=__OUTPUT_DIR__/stdout.txt
#SBATCH --error=__OUTPUT_DIR__/stderr.txt
#SBATCH --partition=main

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)


srun --mpi=pmix --container-image=https://docker/url --container-mounts=/Users/andreyma/workspace/nvidia/cloudai:/work,/Users/andreyma/workspace/nvidia/cloudai:/opt/megatron-lm --no-container-mount-home bash -c "pwd ; ls"