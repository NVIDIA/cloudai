#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH -N 1
#SBATCH --output=__OUTPUT_DIR__/stdout.txt
#SBATCH --error=__OUTPUT_DIR__/stderr.txt
#SBATCH --partition=main

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)


srun --mpi=pmix sleep 5
