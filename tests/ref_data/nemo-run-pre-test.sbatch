#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH -N 1
#SBATCH --output=__OUTPUT_DIR__/output/stdout.txt
#SBATCH --error=__OUTPUT_DIR__/output/stderr.txt
#SBATCH --partition=main

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)


srun --output=__OUTPUT_DIR__/output/pre_test/nccl/stdout.txt --error=__OUTPUT_DIR__/output/pre_test/nccl/stderr.txt --mpi=pmix --container-image=nvcr.io/nvidia/pytorch:24.02-py3 /usr/local/bin/all_reduce_perf_mpi --nthreads 1 --ngpus 1 --minbytes 32M --maxbytes 32M --stepbytes 1M --op sum --datatype float --root 0 --iters 20 --warmup_iters 5 --agg_iters 1 --average 1 --parallel_init 0 --check 1 --blocking 0 --cudagraph 0
SUCCESS_0=$(grep -q "Avg bus bandwidth" __OUTPUT_DIR__/output/pre_test/nccl/stdout.txt && echo 1 || echo 0)
PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && echo 1 || echo 0 )
if [ $PRE_TEST_SUCCESS -eq 1 ]; then
    srun --mpi=pmix --container-image=nvcr.io/nvidia/nemo:24.09 nemo llm pretrain --factory llama_3b -y trainer.num_nodes=1
fi
