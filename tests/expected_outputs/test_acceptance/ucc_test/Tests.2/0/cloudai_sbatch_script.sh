
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