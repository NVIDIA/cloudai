# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from cloudai.systems.slurm import SlurmCommandGenStrategy
from cloudai.util import parse_time_limit

from .nixl_ep import NixlEPCmdArgs, NixlEPTestDefinition


@dataclass(frozen=True)
class NixlEPLaunchWave:
    """Concrete worker launches that should begin after a given phase completes."""

    trigger_phase: int | None
    per_node_processes: tuple[int, ...]


class NixlEPSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for the NIXL Elastic EP benchmark."""

    GENERATED_PLAN_FILE_NAME = "nixl-ep-plan.json"

    @property
    def tdef(self) -> NixlEPTestDefinition:
        return cast(NixlEPTestDefinition, self.test_run.test)

    def image_path(self) -> str | None:
        return str(self.tdef.docker_image.installed_path)

    def _container_mounts(self) -> list[str]:
        return []

    @property
    def processes_per_node(self) -> list[int]:
        raw = self.tdef.cmd_args.num_processes_per_node
        num_nodes, _ = self.get_cached_nodes_spec()
        if isinstance(raw, int):
            return [raw] * num_nodes
        if len(raw) != num_nodes:
            raise ValueError(
                f"num_processes_per_node length ({len(raw)}) must match allocated node count ({num_nodes})"
            )
        return list(raw)

    def _append_sbatch_directives(self, batch_script_content: list[str]) -> None:
        super()._append_sbatch_directives(batch_script_content)
        batch_script_content.extend(
            [
                "",
                "nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )",
                "nodes_array=($nodes)",
                "master_node=${nodes_array[0]}",
                "master_ip=$(srun --nodes=1 --ntasks=1 -w \"$master_node\" hostname --ip-address | awk '{print $1}')",
                "",
                "echo Nodes: $SLURM_JOB_NODELIST",
                "echo Num Nodes: ${#nodes[@]}",
                "echo Master Node: $master_node",
                "echo Master IP: $master_ip",
                "",
            ]
        )

    @property
    def generated_plan_path(self) -> Path:
        return self.test_run.output_path / self.GENERATED_PLAN_FILE_NAME

    def resolve_plan_path(self) -> str:
        self.generated_plan_path.parent.mkdir(parents=True, exist_ok=True)
        self.generated_plan_path.write_text(
            json.dumps(self.inline_plan, indent=2) + "\n",
            encoding="utf-8",
        )
        return str(self.generated_plan_path.absolute())

    @property
    def inline_plan(self) -> list[list[int]]:
        return self.tdef.cmd_args.parse_plan()

    @property
    def phase_transition_timeout_seconds(self) -> int:
        if self.test_run.time_limit:
            return max(int(parse_time_limit(self.test_run.time_limit).total_seconds()), 1)
        return 600

    @property
    def scalar_launch_waves(self) -> list[NixlEPLaunchWave]:
        raw = self.tdef.cmd_args.num_processes_per_node
        num_nodes, _ = self.get_cached_nodes_spec()
        if not isinstance(raw, int):
            return []

        wave_totals = self.tdef.cmd_args.launch_wave_totals()
        if num_nodes == 1:
            expected_total_processes = sum(num_processes for _, num_processes in wave_totals)
            if raw != expected_total_processes:
                raise ValueError(
                    "For single-node NIXL EP runs, num_processes_per_node must match the plan-derived "
                    f"launch waves total ({expected_total_processes}), got {raw}."
                )
            return [NixlEPLaunchWave(trigger_phase, (num_processes,)) for trigger_phase, num_processes in wave_totals]

        total_requested_processes = sum(total for _, total in wave_totals)
        total_capacity = raw * num_nodes
        if total_requested_processes > total_capacity:
            raise ValueError(
                "For multi-node scalar NIXL EP runs, the scalar num_processes_per_node defines the maximum "
                f"number of workers each node can launch across all waves. The plan requires "
                f"{total_requested_processes} total workers, but {num_nodes} nodes with capacity {raw} only "
                f"provide {total_capacity}."
            )

        # Match the upstream multi-node examples by packing each launch wave onto
        # earlier nodes first, only spilling onto later nodes when needed.
        remaining_capacity = [raw] * num_nodes
        packed_waves: list[NixlEPLaunchWave] = []
        for trigger_phase, wave_total in wave_totals:
            remaining_wave_total = wave_total
            per_node_wave_sizes = [0] * num_nodes
            for node_idx in range(num_nodes):
                if remaining_wave_total == 0:
                    break
                assignable = min(remaining_capacity[node_idx], remaining_wave_total)
                per_node_wave_sizes[node_idx] = assignable
                remaining_capacity[node_idx] -= assignable
                remaining_wave_total -= assignable

            if remaining_wave_total != 0:
                raise ValueError(
                    "For multi-node scalar NIXL EP runs, the plan-derived launch waves cannot be packed onto "
                    f"{num_nodes} nodes with per-node capacity {raw}. Remaining wave size: {remaining_wave_total}."
                )

            packed_waves.append(NixlEPLaunchWave(trigger_phase, tuple(per_node_wave_sizes)))

        return packed_waves

    def build_elastic_command(self, num_processes: int, include_tcp_server: bool = False) -> list[str]:
        cmd_args: NixlEPCmdArgs = self.tdef.cmd_args
        command = [
            cmd_args.python_executable,
            cmd_args.elastic_script,
            "--plan",
            self.resolve_plan_path(),
            "--num-processes",
            str(num_processes),
        ]

        if include_tcp_server:
            command.extend(["--tcp-server", "$master_ip"])
        for arg, value in sorted((cmd_args.model_extra or {}).items()):
            flag = "--" + arg.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            elif value is not None:
                command.extend([flag, str(value)])

        return command

    def generate_wait_for_master_services_function(self) -> str:
        # The upstream rank server assigns a rank on any successful TCP
        # connection, so the readiness probe must not touch that port.
        return f"""\
wait_for_master_services() {{
    local timeout={self.tdef.cmd_args.service_startup_timeout_seconds}
    local interval=1
    local end_time=$(($(date +%s) + timeout))

    while [ "$(date +%s)" -lt "$end_time" ]; do
        if timeout 1 bash -c ": > /dev/tcp/$master_ip/{self.tdef.cmd_args.store_port}" >/dev/null 2>&1; then
            echo "NIXL EP master services are ready on $master_ip"
            return 0
        fi
        sleep "$interval"
    done

    echo "Timed out waiting for NIXL EP master services on $master_ip"
    return 1
}}"""

    def _launch_srun_prefix(self, node_idx: int) -> str:
        target_arg = "--nodelist=$SLURM_JOB_MASTER_NODE" if node_idx == 0 else f"--relative={node_idx}"
        parts = [
            *self.gen_srun_prefix(with_num_nodes=False),
            "--overlap",
            target_arg,
            "--ntasks-per-node=1",
            "--ntasks=1",
            "-N1",
        ]
        return " ".join(parts)

    def _launch_srun_command(
        self,
        node_idx: int,
        num_processes: int,
        *,
        include_tcp_server: bool | None = None,
        append_output: bool = False,
    ) -> str:
        if include_tcp_server is None:
            include_tcp_server = node_idx != 0
        command = " ".join(self.build_elastic_command(num_processes, include_tcp_server=include_tcp_server))
        env_file = (self.test_run.output_path / "env_vars.sh").absolute()
        log_file = (self.test_run.output_path / f"nixl-ep-node-{node_idx}.log").absolute()
        open_mode_arg = " --open-mode=append" if append_output else ""
        script = f"source {shlex.quote(str(env_file))}; {command}".replace('"', '\\"')
        return f'{self._launch_srun_prefix(node_idx)}{open_mode_arg} --output={log_file} bash -c "{script}"'

    def generate_wait_for_phase_completion_function(self) -> str:
        timeout = self.phase_transition_timeout_seconds
        return f"""\
wait_for_phase_completion() {{
    local phase="$1"
    local log_file="$2"
    local primary_pid="$3"
    local timeout={timeout}
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
}}"""

    def _write_env_vars_file(self) -> None:
        self.test_run.output_path.mkdir(parents=True, exist_ok=True)
        with (self.test_run.output_path / "env_vars.sh").open("w") as env_file:
            for key, value in self.final_env_vars.items():
                env_file.write(f"export {key}={value}\n")

    def _append_background_launch(
        self,
        lines: list[str],
        node_idx: int,
        num_processes: int,
        *,
        include_tcp_server: bool | None = None,
        append_output: bool = False,
    ) -> None:
        lines.append(
            self._launch_srun_command(
                node_idx,
                num_processes,
                include_tcp_server=include_tcp_server,
                append_output=append_output,
            )
            + " &"
        )
        lines.append("worker_pids+=($!)")

    @staticmethod
    def _append_wait_for_workers(lines: list[str]) -> None:
        lines.extend(
            [
                "",
                "rc=0",
                'for pid in "${worker_pids[@]}"; do',
                '    wait "$pid" || rc=$?',
                "done",
                "",
                "exit $rc",
            ]
        )

    def _gen_scalar_wave_command(self, waves: list[NixlEPLaunchWave]) -> str:
        num_nodes, _ = self.get_cached_nodes_spec()
        primary_log_file = (self.test_run.output_path / "nixl-ep-node-0.log").absolute()
        initial_wave = waves[0]
        lines: list[str] = []
        if num_nodes > 1:
            lines.extend([self.generate_wait_for_master_services_function(), ""])
        if len(waves) > 1:
            lines.extend([self.generate_wait_for_phase_completion_function(), ""])

        lines.extend(
            [
                "worker_pids=()",
                "",
                'echo "Starting initial NIXL EP wave on the master node..."',
                self._launch_srun_command(0, initial_wave.per_node_processes[0]) + " &",
                "primary_pid=$!",
                "worker_pids+=($primary_pid)",
            ]
        )

        if num_nodes > 1:
            lines.extend(
                [
                    "",
                    'echo "Waiting for NIXL EP master services..."',
                    "wait_for_master_services || exit 1",
                ]
            )
            follower_initial_wave = initial_wave.per_node_processes[1:]
            if any(num_processes > 0 for num_processes in follower_initial_wave):
                lines.extend(["", 'echo "Starting initial NIXL EP wave on follower nodes..."'])
                for node_idx, num_processes in enumerate(follower_initial_wave, start=1):
                    if num_processes <= 0:
                        continue
                    self._append_background_launch(lines, node_idx, num_processes)

        for wave_idx, wave in enumerate(waves[1:], start=1):
            scope = "across allocated nodes" if num_nodes > 1 else "on the master node"
            lines.extend(
                [
                    "",
                    f'echo "Waiting for phase {wave.trigger_phase} before starting wave {wave_idx}..."',
                    f'wait_for_phase_completion "{wave.trigger_phase}" "{primary_log_file}" "$primary_pid" || exit 1',
                    f'echo "Starting NIXL EP wave {wave_idx} {scope}..."',
                ]
            )
            for node_idx, num_processes in enumerate(wave.per_node_processes):
                if num_processes <= 0:
                    continue
                self._append_background_launch(
                    lines,
                    node_idx,
                    num_processes,
                    include_tcp_server=True,
                    append_output=True,
                )

        self._append_wait_for_workers(lines)
        return "\n".join(lines)

    def _gen_fixed_process_layout_command(self, processes_per_node: list[int]) -> str:
        if len(processes_per_node) == 1:
            return "\n".join(
                [
                    'echo "Starting NIXL EP on the master node..."',
                    self._launch_srun_command(0, processes_per_node[0]),
                ]
            )

        lines = [
            self.generate_wait_for_master_services_function(),
            "",
            "worker_pids=()",
            "",
            'echo "Starting NIXL EP on the master node..."',
            self._launch_srun_command(0, processes_per_node[0]) + " &",
            "worker_pids+=($!)",
            "",
            'echo "Waiting for NIXL EP master services..."',
            "wait_for_master_services || exit 1",
            "",
            'echo "Starting NIXL EP follower nodes..."',
        ]

        for idx, num_processes in enumerate(processes_per_node[1:], start=1):
            lines.append(self._launch_srun_command(idx, num_processes) + " &")
            lines.append("worker_pids+=($!)")

        self._append_wait_for_workers(lines)
        return "\n".join(lines)

    def _gen_srun_command(self) -> str:
        self._write_env_vars_file()
        processes_per_node = self.processes_per_node

        if isinstance(self.tdef.cmd_args.num_processes_per_node, int):
            waves = self.scalar_launch_waves
            if len(processes_per_node) == 1 and len(waves) <= 1:
                single_wave_processes = waves[0].per_node_processes[0]
                return "\n".join(
                    [
                        'echo "Starting NIXL EP on the master node..."',
                        self._launch_srun_command(0, single_wave_processes),
                    ]
                )
            return self._gen_scalar_wave_command(waves)

        return self._gen_fixed_process_layout_command(processes_per_node)
