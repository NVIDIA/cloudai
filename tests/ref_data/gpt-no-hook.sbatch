#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --partition=main
#SBATCH -N 1

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export COMBINE_THRESHOLD=1
export PER_GPU_COMBINE_THRESHOLD=0
export XLA_FLAGS="--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD --xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD --xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD"
srun --export=ALL --mpi=pmix --container-image=https://docker/url --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/install:/cloudai_install,__OUTPUT_DIR__/output,__OUTPUT_DIR__/output:/opt/paxml/workspace/ --output=__OUTPUT_DIR__/output/mapping-stdout.txt --error=__OUTPUT_DIR__/output/mapping-stderr.txt bash -c "echo \$(date): \$(hostname):node \${SLURM_NODEID}:rank \${SLURM_PROCID}."

srun --export=ALL --mpi=pmix --container-image=https://docker/url --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/install:/cloudai_install,__OUTPUT_DIR__/output,__OUTPUT_DIR__/output:/opt/paxml/workspace/ --ntasks=1 --ntasks-per-node=1 --output=__OUTPUT_DIR__/output/metadata/node-%N.toml --error=__OUTPUT_DIR__/output/metadata/nodes.err bash /cloudai_install/slurm-metadata.sh

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
    --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/install:/cloudai_install,__OUTPUT_DIR__/output,__OUTPUT_DIR__/output:/opt/paxml/workspace/ \
    /opt/paxml/workspace/run.sh
