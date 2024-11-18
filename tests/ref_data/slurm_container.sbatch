#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH -N 1
#SBATCH --output=__OUTPUT_DIR__/output/stdout.txt
#SBATCH --error=__OUTPUT_DIR__/output/stderr.txt
#SBATCH --partition=main

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)


srun --mpi=pmix --container-image=https://docker/url --container-mounts=__OUTPUT_DIR__/install/url__commit_hash:/work,__OUTPUT_DIR__/install/repo__mcore_vfm_commit_hash:/opt/megatron-lm --no-container-mount-home bash -c "pwd ; ls"
