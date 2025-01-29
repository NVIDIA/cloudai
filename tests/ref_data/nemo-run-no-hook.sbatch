#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH -N 1
#SBATCH --output=__OUTPUT_DIR__/output/stdout.txt
#SBATCH --error=__OUTPUT_DIR__/output/stderr.txt
#SBATCH --partition=main

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)


srun --mpi=pmix --container-image=nvcr.io/nvidia/nemo:24.09 nemo llm pretrain --factory llama_3b -y trainer.num_nodes=1 trainer.max_steps=5 trainer.val_check_interval=1000 log.ckpt.save_on_train_epoch_end=False log.ckpt.save_last=False
