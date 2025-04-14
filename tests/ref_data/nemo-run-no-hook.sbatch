#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH -N 1
#SBATCH --output=__OUTPUT_DIR__/output/stdout.txt
#SBATCH --error=__OUTPUT_DIR__/output/stderr.txt
#SBATCH --partition=main

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export CLOUDAI_NEMO_RECIPE=llama_3b
export CLOUDAI_NEMO_TASK=pretrain
srun --export=ALL --mpi=pmix --container-image=nvcr.io/nvidia/nemo:24.09 --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__CLOUDAI_DIR__/src/cloudai/workloads/nemo_run:/cloudai_workspace,__OUTPUT_DIR__/install:/cloudai_install --output=__OUTPUT_DIR__/output/mapping-stdout.txt --error=__OUTPUT_DIR__/output/mapping-stderr.txt bash -c "echo \$(date): \$(hostname):node \${SLURM_NODEID}:rank \${SLURM_PROCID}."

srun --export=ALL --mpi=pmix --container-image=nvcr.io/nvidia/nemo:24.09 --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__CLOUDAI_DIR__/src/cloudai/workloads/nemo_run:/cloudai_workspace,__OUTPUT_DIR__/install:/cloudai_install --ntasks=1 --ntasks-per-node=1 --output=__OUTPUT_DIR__/output/metadata/node-%N.toml --error=__OUTPUT_DIR__/output/metadata/nodes.err bash /cloudai_install/slurm-metadata.sh

srun --export=ALL --mpi=pmix --container-image=nvcr.io/nvidia/nemo:24.09 --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__CLOUDAI_DIR__/src/cloudai/workloads/nemo_run:/cloudai_workspace,__OUTPUT_DIR__/install:/cloudai_install python /cloudai_install/cloudai_nemorun.py --factory llama_3b -y trainer.max_steps=100 trainer.val_check_interval=1000 trainer.num_nodes=1 trainer.strategy.tensor_model_parallel_size=1 trainer.strategy.pipeline_model_parallel_size=1 trainer.strategy.context_parallel_size=2 data.seq_length=8192 data.micro_batch_size=1 data.global_batch_size=1