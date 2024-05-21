
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