#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH -N 1
#SBATCH --output=__OUTPUT_DIR__/stdout.txt
#SBATCH --error=__OUTPUT_DIR__/stderr.txt
#SBATCH --partition=main

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)


srun --mpi=pmix --container-image=nvcr.io/nvidia/pytorch:24.02-py3 /opt/hpcx/ucc/bin/ucc_perftest -c alltoall -b 1 -e 8M -m cuda -F
