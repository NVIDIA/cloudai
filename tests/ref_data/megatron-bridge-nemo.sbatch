#!/bin/bash
# Representative NeMo Run Slurm template. Paths are substituted by the acceptance test.
#SBATCH --job-name=nemo-training
#SBATCH --output=__OUTPUT__/nemo.out
#SBATCH --error=__OUTPUT__/nemo.err
#SBATCH --nodes=99
#SBATCH --nodelist=upstream-node
#SBATCH --partition=upstream-partition
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --open-mode=append

export PRIORITY=upstream
export HF_TOKEN=dummy_token
export UPSTREAM_ONLY=retained
export PYTHONPATH=/opt/Megatron-Bridge
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
set +e
srun --output=__OUTPUT__/experiments/run/log_%j.out --container-image=__IMAGE__ bash -c 'source /cloudai_run_results/env_vars.sh; __PROFILE__ python train.py'
exit $?
