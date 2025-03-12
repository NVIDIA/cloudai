#!/bin/bash
#SBATCH --job-name=ray_docker
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt

# Get node information
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
head_node=${nodes_array[0]}
port=6379


source /nethome/aagrawal360/mambaforge/etc/profile.d/conda.sh
source /nethome/aagrawal360/mambaforge/etc/profile.d/mamba.sh
mamba activate /home/aagrawal360/repos/cloudai/env

if [ "$SLURMD_NODENAME" == "$head_node" ]; then
    echo "Starting Ray head node on $head_node"
    ray start --head --port=$port --num-cpus=$SLURM_CPUS_PER_TASK
    sleep 30
    python /home/aagrawal360/repos/cloudai/experiment/ray_test/ray_test_job.py
    ray status
else
    echo "Starting Ray worker on $SLURMD_NODENAME"
    ray start --address=${head_node}:${port} --num-cpus=$SLURM_CPUS_PER_TASK
fi

sleep 30