#!/bin/bash

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=("${nodes[@]:0:3}")
master_node=${nodes_array[0]}
export SLURM_JOB_MASTER_NODE="${SLURM_JOB_MASTER_NODE:-$master_node}"
srun --nodes=1 --ntasks=1 -N1 --nodelist="$master_node" --output=__OUTPUT_DIR__/output/nixl-ep-master-ip.txt --error=__OUTPUT_DIR__/output/stderr.txt hostname --ip-address
master_ip=$(awk '{print $1}' __OUTPUT_DIR__/output/nixl-ep-master-ip.txt)

echo "Nodes: $SLURM_JOB_NODELIST"
echo "Num Nodes: ${#nodes_array[@]}"
echo "Master Node: $master_node"
echo "Master IP: $master_ip"

wait_for_launch_step_id() {
    local step_name="$1"
    local step_id
    for _ in {1..10}; do
        step_id=$(squeue --noheader --steps --job "$SLURM_JOB_ID" --format="%i %j" | awk -v step_name="$step_name" '$2 == step_name { print $1; exit }')
        if [ -n "$step_id" ]; then
            echo "$step_id"
            return 0
        fi
        sleep 1
    done
    return 1
}

signal_nixl_ep_steps() {
    local signal="$1"
    local idx
    local step_id
    for idx in "${!launch_step_names[@]}"; do
        step_id="${launch_step_ids[$idx]:-}"
        if [ -z "$step_id" ]; then
            step_id="$(wait_for_launch_step_id "${launch_step_names[$idx]}" || true)"
            launch_step_ids[$idx]="$step_id"
        fi
        if [ -n "$step_id" ]; then
            scancel --signal="$signal" "$step_id" >/dev/null 2>&1 || true
        fi
    done
}

cleanup_nixl_ep() {
    local pids
    pids="$(jobs -pr)"
    if [ -z "$pids" ]; then
        return 0
    fi
    echo "Cleaning up NIXL EP background launches..."
    signal_nixl_ep_steps "TERM"
    sleep 2
    pids="$(jobs -pr)"
    if [ -n "$pids" ]; then
        signal_nixl_ep_steps "KILL"
    fi
    wait >/dev/null 2>&1 || true
}

on_nixl_ep_signal() {
    local rc="$1"
    echo "NIXL EP launcher received signal, rc=$rc"
    cleanup_nixl_ep
    echo "NIXL EP launcher exiting with rc=$rc"
    exit "$rc"
}

trap cleanup_nixl_ep EXIT
trap 'on_nixl_ep_signal 130' INT
trap 'on_nixl_ep_signal 143' TERM

wait_for_master_services() {
    local timeout=90
    local interval=1
    local end_time=$(($(date +%s) + timeout))

    while [ "$(date +%s)" -lt "$end_time" ]; do
        if timeout 1 bash -c ": > /dev/tcp/$master_ip/9999" >/dev/null 2>&1; then
            echo "NIXL EP master services are ready on $master_ip"
            return 0
        fi
        sleep "$interval"
    done

    echo "Timed out waiting for NIXL EP master services on $master_ip"
    return 1
}

wait_for_phase_completion() {
    local phase="$1"
    local log_file="$2"
    local primary_pid="$3"
    local timeout=150
    local interval=1
    local end_time=$(($(date +%s) + timeout))

    while [ "$(date +%s)" -lt "$end_time" ]; do
        if [ -f "$log_file" ] && grep -Fq -- "-> end phase $phase" "$log_file"; then
            echo "Detected completion of phase $phase in $log_file"
            return 0
        fi
        if [ -f "$log_file" ] && grep -Fq -- "no plan phases were found for rank" "$log_file"; then
            echo "Detected an early NIXL EP failure while waiting for phase $phase"
            return 1
        fi
        if ! kill -0 "$primary_pid" >/dev/null 2>&1; then
            echo "Primary NIXL EP launch exited before phase $phase completed"
            return 1
        fi
        sleep "$interval"
    done

    echo "Timed out waiting for phase $phase to complete"
    return 1
}

launch_pids=()
launch_step_names=()
launch_step_ids=()
launch_expected_removal=()

echo "Starting initial NIXL EP stage on the master node..."
launch_step_names+=( "nixl-ep-p0-l0" )
srun --export=ALL --mpi=pmix --container-image=docker.io/nvidia/nixl-ep:latest --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__INSTALL_DIR__:/cloudai_install,__OUTPUT_DIR__/output --overlap --job-name=nixl-ep-p0-l0 --nodelist="${nodes_array[0]}" --ntasks-per-node=1 --ntasks=1 -N1 --output=__OUTPUT_DIR__/output/nixl-ep-node-0.log --error=__OUTPUT_DIR__/output/nixl-ep-node-0.log bash -c "source __OUTPUT_DIR__/output/env_vars.sh; python3 /workspace/nixl/examples/device/ep/tests/elastic/elastic.py --plan __OUTPUT_DIR__/output/nixl-ep-plan.json --num-processes 4 --disable-ll-nvlink --hidden-dim 8192 --kineto --num-experts-per-rank 4 --num-tokens 256 --num-topk 6" &
primary_pid=$!
launch_pids+=( "$primary_pid" )
launch_step_id="$(wait_for_launch_step_id "nixl-ep-p0-l0" || true)"
echo "NIXL EP launch nixl-ep-p0-l0 step ID: ${launch_step_id:-unknown}"
launch_step_ids+=( "$launch_step_id" )
launch_expected_removal+=( "0" )

echo "Waiting for NIXL EP master services..."
wait_for_master_services || exit 1

echo "Waiting for phase 0 before starting phase 1..."
wait_for_phase_completion "0" "__OUTPUT_DIR__/output/nixl-ep-node-0.log" "$primary_pid" || exit 1

echo "Starting launches for phase 1..."
launch_step_names+=( "nixl-ep-p1-l0" )
srun --export=ALL --mpi=pmix --container-image=docker.io/nvidia/nixl-ep:latest --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__INSTALL_DIR__:/cloudai_install,__OUTPUT_DIR__/output --overlap --job-name=nixl-ep-p1-l0 --nodelist="${nodes_array[1]}" --ntasks-per-node=1 --ntasks=1 -N1 --open-mode=append --output=__OUTPUT_DIR__/output/nixl-ep-node-1.log --error=__OUTPUT_DIR__/output/nixl-ep-node-1.log bash -c "source __OUTPUT_DIR__/output/env_vars.sh; python3 /workspace/nixl/examples/device/ep/tests/elastic/elastic.py --plan __OUTPUT_DIR__/output/nixl-ep-plan.json --num-processes 4 --tcp-server $master_ip --disable-ll-nvlink --hidden-dim 8192 --kineto --num-experts-per-rank 4 --num-tokens 256 --num-topk 6" &
launch_pids+=( "$!" )
launch_step_id="$(wait_for_launch_step_id "nixl-ep-p1-l0" || true)"
echo "NIXL EP launch nixl-ep-p1-l0 step ID: ${launch_step_id:-unknown}"
launch_step_ids+=( "$launch_step_id" )
launch_expected_removal+=( "1" )

echo "Waiting for phase 2 before starting phase 3..."
wait_for_phase_completion "2" "__OUTPUT_DIR__/output/nixl-ep-node-0.log" "$primary_pid" || exit 1

echo "Starting launches for phase 3..."
launch_step_names+=( "nixl-ep-p3-l0" )
srun --export=ALL --mpi=pmix --container-image=docker.io/nvidia/nixl-ep:latest --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__INSTALL_DIR__:/cloudai_install,__OUTPUT_DIR__/output --overlap --job-name=nixl-ep-p3-l0 --nodelist="${nodes_array[2]}" --ntasks-per-node=1 --ntasks=1 -N1 --open-mode=append --output=__OUTPUT_DIR__/output/nixl-ep-node-2.log --error=__OUTPUT_DIR__/output/nixl-ep-node-2.log bash -c "source __OUTPUT_DIR__/output/env_vars.sh; python3 /workspace/nixl/examples/device/ep/tests/elastic/elastic.py --plan __OUTPUT_DIR__/output/nixl-ep-plan.json --num-processes 2 --tcp-server $master_ip --disable-ll-nvlink --hidden-dim 8192 --kineto --num-experts-per-rank 4 --num-tokens 256 --num-topk 6" &
launch_pids+=( "$!" )
launch_step_id="$(wait_for_launch_step_id "nixl-ep-p3-l0" || true)"
echo "NIXL EP launch nixl-ep-p3-l0 step ID: ${launch_step_id:-unknown}"
launch_step_ids+=( "$launch_step_id" )
launch_expected_removal+=( "0" )

pending_term=0
ignored_expected_removal=0
on_nixl_ep_deferred_term() {
    pending_term=1
    echo "Deferring TERM while collecting NIXL EP launch statuses"
}
trap on_nixl_ep_deferred_term TERM

rc=0
for idx in "${!launch_pids[@]}"; do
    wait "${launch_pids[$idx]}"
    wait_rc=$?
    if [ "$wait_rc" -eq 0 ]; then
        continue
    fi
    if [ "${launch_expected_removal[$idx]}" = "1" ] && [ "$wait_rc" -eq 143 ]; then
        echo "Ignoring expected NIXL EP planned-rank-removal exit from launch PID ${launch_pids[$idx]}"
        ignored_expected_removal=1
    elif [ "$pending_term" -ne 0 ] && [ "$wait_rc" -eq 143 ]; then
        echo "Ignoring deferred TERM wait interruption for launch PID ${launch_pids[$idx]}"
    elif [ "$rc" -eq 0 ]; then
        rc=$wait_rc
    fi
done

if [ "$pending_term" -ne 0 ] && [ "$ignored_expected_removal" -eq 0 ] && [ "$rc" -eq 0 ]; then
    rc=143
fi

if [ "$rc" -eq 0 ]; then
    echo "All NIXL EP launches completed successfully"
fi

echo "NIXL EP launcher exiting with rc=$rc"
exit "$rc"
