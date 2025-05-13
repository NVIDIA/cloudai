#!/bin/bash

export NIM_LEADER_IP_ADDRESS=${SLURM_JOB_MASTER_NODE}
export NIM_NODE_RANK=${SLURM_NODEID}

export NIM_MODEL_NAME='__OUTPUT_DIR__/output'
export NIM_CACHE_PATH='__OUTPUT_DIR__/output'

if [ "$NIM_NODE_RANK" -eq 0 ]; then
  export NIM_LEADER_ROLE=1
else
  export NIM_LEADER_ROLE=0
fi

echo "Starting NIM server on node rank ${NIM_NODE_RANK} with leader role ${NIM_LEADER_ROLE}"
exec /opt/nim/start_server.sh