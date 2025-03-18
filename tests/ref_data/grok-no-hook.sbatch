#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH -N 1
#SBATCH --partition=main

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export COMBINE_THRESHOLD=1
export PER_GPU_COMBINE_THRESHOLD=0
export XLA_FLAGS="--xla_disable_hlo_passes=rematerialization --xla_dump_hlo_pass_re=.* --xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD --xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_highest_priority_async_stream=true --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_reduce_scatter_combine_by_dim=false --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_graph_level=0 --xla_gpu_pgle_profile_file_or_directory_path=/opt/paxml/workspace/pgle_output_profile.pbtxt --xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD --xla_gpu_run_post_layout_collective_pipeliner=false --xla_gpu_use_memcpy_local_p2p=false"
srun --mpi=pmix --container-image=https://docker/url --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/output:/opt/paxml/workspace/,__OUTPUT_DIR__/install:/cloudai_install --output=__OUTPUT_DIR__/output/mapping-stdout.txt --error=__OUTPUT_DIR__/output/mapping-stderr.txt bash -c "echo \$(date): \$(hostname):node \${SLURM_NODEID}:rank \${SLURM_PROCID}."

srun --mpi=pmix --container-image=https://docker/url --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/output:/opt/paxml/workspace/,__OUTPUT_DIR__/install:/cloudai_install --ntasks-per-node=1 --output=__OUTPUT_DIR__/output/metadata/node-%N.toml --error=__OUTPUT_DIR__/output/metadata/nodes.err bash /cloudai_install/slurm-metadata.sh

echo "Loading container with srun command"
    srun --mpi=none --container-image=https://docker/url --container-name=cont true
    echo "Running srun command"
    srun \
    --mpi=none \
     \
    --export=ALL \
    -o __OUTPUT_DIR__/output/output-%j-%n-%t.txt \
    -e __OUTPUT_DIR__/output/error-%j-%n-%t.txt \
    --container-name=cont \
    --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/output:/opt/paxml/workspace/,__OUTPUT_DIR__/install:/cloudai_install \
    /opt/paxml/workspace/run.sh
