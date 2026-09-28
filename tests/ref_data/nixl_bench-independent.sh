#!/bin/bash
set -e
nixl_output=__OUTPUT_DIR__/output/nixlbench
nixl_task="$SLURM_PROCID"
trap 'echo "$?" > "$nixl_output/$nixl_task.status"' EXIT
exec > "$nixl_output/$nixl_task.stdout" 2> "$nixl_output/$nixl_task.stderr"
hostname > "$nixl_output/$nixl_task.hostname"
if [ "$nixl_task" -eq 0 ]; then
    echo "$SLURM_NTASKS" > "$nixl_output/ntasks"
fi
source __OUTPUT_DIR__/output/env_vars.sh
./nixlbench --backend=POSIX
