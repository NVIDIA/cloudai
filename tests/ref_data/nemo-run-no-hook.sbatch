#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH -N 1
#SBATCH --output=__OUTPUT_DIR__/output/stdout.txt
#SBATCH --error=__OUTPUT_DIR__/output/stderr.txt
#SBATCH --partition=main

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)


srun --mpi=pmix --container-image=nvcr.io/nvidia/nemo:24.09 --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__CLOUDAI_DIR__/src/cloudai/workloads/nemo_run:/cloudai_workspace,__OUTPUT_DIR__/output/nemorun-workspace:/workspace python /cloudai_workspace/cloudai_nemorun.py --factory llama_3b -y trainer.max_steps=100 trainer.val_check_interval=1000 trainer.num_nodes=1 trainer.strategy.tensor_model_parallel_size=1 trainer.strategy.pipeline_model_parallel_size=1 trainer.strategy.context_parallel_size=2 log.ckpt.save_on_train_epoch_end=False log.ckpt.save_last=False data.micro_batch_size=1 data.global_batch_size=1