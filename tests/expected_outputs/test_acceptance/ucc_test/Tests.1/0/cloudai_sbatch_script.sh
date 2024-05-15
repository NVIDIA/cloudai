#!/bin/bash
#SBATCH --job-name=alltoall_20240520_171139
#SBATCH -N 2
#SBATCH --output=/labhome/eshukrun/tmp/pytest-of-eshukrun/pytest-25/test_slurm_conf_v0_6_general_t2/2024-05-20_17-11-39/Tests.1/0/stdout.txt
#SBATCH --error=/labhome/eshukrun/tmp/pytest-of-eshukrun/pytest-25/test_slurm_conf_v0_6_general_t2/2024-05-20_17-11-39/Tests.1/0/stderr.txt
#SBATCH --partition=partition_1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MELLANOX_VISIBLE_DEVICES=0,3,4,5,6,9,10,11
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TIMEOUT=20

srun \
--mpi=pmix \
--container-image=/path/to/install/ucc-test/ucc_test.sqsh \
/opt/hpcx/ucc/bin/ucc_perftest \
-c alltoall \
-b 1 \
-e 8M \
-m cuda \
-F