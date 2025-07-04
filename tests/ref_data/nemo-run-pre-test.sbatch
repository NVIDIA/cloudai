#!/bin/bash
# generated by CloudAI@__CLOUDAI_VERSION__
#SBATCH --job-name=__JOB_NAME__
#SBATCH --output=__OUTPUT_DIR__/output/stdout.txt
#SBATCH --error=__OUTPUT_DIR__/output/stderr.txt
#SBATCH --partition=main
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH --gres=gpu:8

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export CLOUDAI_NEMO_RECIPE=llama_3b
export CLOUDAI_NEMO_TASK=pretrain
srun --export=ALL --mpi=pmix --container-image=nvcr.io/nvidia/nemo:24.09 --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/install:/cloudai_install,__OUTPUT_DIR__/output,__CLOUDAI_DIR__/src/cloudai/workloads/nemo_run:/cloudai_workspace --output=__OUTPUT_DIR__/output/mapping-stdout.txt --error=__OUTPUT_DIR__/output/mapping-stderr.txt bash -c "echo \$(date): \$(hostname):node \${SLURM_NODEID}:rank \${SLURM_PROCID}."

srun --export=ALL --mpi=pmix --container-image=nvcr.io/nvidia/nemo:24.09 --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/install:/cloudai_install,__OUTPUT_DIR__/output,__CLOUDAI_DIR__/src/cloudai/workloads/nemo_run:/cloudai_workspace --ntasks=1 --ntasks-per-node=1 --output=__OUTPUT_DIR__/output/metadata/node-%N.toml --error=__OUTPUT_DIR__/output/metadata/nodes.err bash /cloudai_install/slurm-metadata.sh

srun --output=__OUTPUT_DIR__/output/pre_test/nccl/stdout.txt --error=__OUTPUT_DIR__/output/pre_test/nccl/stderr.txt --export=ALL --mpi=pmix --container-image=nvcr.io/nvidia/pytorch:24.02-py3 --container-mounts=__OUTPUT_DIR__/output/pre_test/nccl:/cloudai_run_results,__OUTPUT_DIR__/install:/cloudai_install,__OUTPUT_DIR__/output/pre_test/nccl bash -c "source __OUTPUT_DIR__/output/pre_test/nccl/env_vars.sh; all_reduce_perf_mpi --nthreads 1 --ngpus 1 --minbytes 32M --maxbytes 32M --stepbytes 1M --op sum --datatype float --root 0 --iters 20 --warmup_iters 5 --agg_iters 1 --average 1 --parallel_init 0 --check 1 --blocking 0 --cudagraph 0"
SUCCESS_0=$(grep -q "Avg bus bandwidth" __OUTPUT_DIR__/output/pre_test/nccl/stdout.txt && echo 1 || echo 0)
PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && echo 1 || echo 0 )
if [ $PRE_TEST_SUCCESS -eq 1 ]; then
    srun --export=ALL --mpi=pmix --container-image=nvcr.io/nvidia/nemo:24.09 --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/install:/cloudai_install,__OUTPUT_DIR__/output,__CLOUDAI_DIR__/src/cloudai/workloads/nemo_run:/cloudai_workspace bash -c "source __OUTPUT_DIR__/output/env_vars.sh; python /cloudai_install/cloudai_nemorun.py --factory llama_3b -y trainer.max_steps=100 trainer.val_check_interval=1000 trainer.num_nodes=1 trainer.strategy.tensor_model_parallel_size=1 trainer.strategy.pipeline_model_parallel_size=1 trainer.strategy.context_parallel_size=2 trainer.devices=8 data.seq_length=8192 data.micro_batch_size=1 data.global_batch_size=1 data.num_train_samples=100"
fi