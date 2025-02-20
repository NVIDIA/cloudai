#!/bin/bash
#SBATCH --job-name=coreai_resiliency_osiris-TestTemplate.20250212_122045
#SBATCH -N 1
#SBATCH --output=results/nemo_run_llama3_8b/dse_nemo_run_llama3_8b_1/0/1/stdout.txt
#SBATCH --error=results/nemo_run_llama3_8b/dse_nemo_run_llama3_8b_1/0/1/stderr.txt
#SBATCH --partition=batch
#SBATCH --account=coreai_resiliency_osiris
#SBATCH --gpus-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

srun --mpi=pmix --gpus-per-node=8 --partition=batch --account=coreai_resiliency_osiris --container-image=/lustre/fsw/portfolios/coreai/users/srivatsank/workspace/sri_cloudai/cloudai/install/nvcr.io_nvidia__nemo__24.12.rc3.sqsh --container-mounts=/lustre/fsw/portfolios/coreai/users/srivatsank/workspace/sri_cloudai/cloudai/results/nemo_run_llama3_8b/dse_nemo_run_llama3_8b_1/0/1/:/workspace/ python /workspace/cloudai_nemorun.py --yes --factory llama3_8b log.ckpt.save_last=False data.global_batch_size=32 trainer.val_check_interval=1000 trainer.callbacks="combined_callbacks" trainer.max_steps=100 data.seq_length=4096 trainer.strategy.tensor_model_parallel_size=2 trainer.strategy.context_parallel_size=1