#!/bin/bash
#SBATCH --job-name=all_gather_perf_mpi_20240520_171308
#SBATCH -N 2
#SBATCH --output=/labhome/eshukrun/tmp/pytest-of-eshukrun/pytest-26/test_slurm_conf_v0_6_general_t0/2024-05-20_17-12-47/Tests.22/0/stdout.txt
#SBATCH --error=/labhome/eshukrun/tmp/pytest-of-eshukrun/pytest-26/test_slurm_conf_v0_6_general_t0/2024-05-20_17-12-47/Tests.22/0/stderr.txt
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
export NCCL_TEST_SPLIT_MASK=0x7

srun \
--mpi=pmix \
--container-image=/path/to/install/nccl-test/nccl_test.sqsh \
/usr/local/bin/all_gather_perf_mpi \
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