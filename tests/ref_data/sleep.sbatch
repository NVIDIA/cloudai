#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH -N 1
#SBATCH --output=__OUTPUT_DIR__/output/stdout.txt
#SBATCH --error=__OUTPUT_DIR__/output/stderr.txt
#SBATCH --partition=main

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

srun --mpi=pmix --output=__OUTPUT_DIR__/output/mapping-stdout.txt --error=__OUTPUT_DIR__/output/mapping-stderr.txt bash -c "echo \$(date): \$(hostname):node \${SLURM_NODEID}:rank \${SLURM_PROCID}."

srun --mpi=pmix --ntasks-per-node=1 --output=__OUTPUT_DIR__/output/metadata/node-%N.toml --error=__OUTPUT_DIR__/output/metadata/nodes.err bash /cloudai_install/slurm-system-info.sh

srun --mpi=pmix sleep 5
