#!/bin/bash
#SBATCH --job-name=alltoall_perf_mpi_20240515_120303
#SBATCH -N 2
#SBATCH --output=/.autodirect/mtrsysgwork/eshukrun/cloudai/results/2024-05-15_12-02-40/Tests.24/0/stdout.txt
#SBATCH --error=/.autodirect/mtrsysgwork/eshukrun/cloudai/results/2024-05-15_12-02-40/Tests.24/0/stderr.txt
#SBATCH --partition=partition_1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:20:00

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MELLANOX_VISIBLE_DEVICES=0,3,4,5,6,9,10,11
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TIMEOUT=20

srun \
--mpi=pmix \
--container-image=/path/to/install/nccl-test/nccl_test.sqsh \
/usr/local/bin/alltoall_perf_mpi \
--nthreads 1 \
--ngpus 1 \
--minbytes 128 \
--maxbytes 4G \
--stepbytes 1M \
--op sum \
--datatype float \
--root 0 \
--iters 100 \
--warmup_iters 50 \
--agg_iters 1 \
--average 1 \
--parallel_init 0 \
--check 1 \
--blocking 0 \
--cudagraph 0 \
--stepfactor 2