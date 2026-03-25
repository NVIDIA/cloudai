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

from .nixl_ep import GENERATED_PLAN_FILE_NAME, NixlEPCmdArgs, NixlEPTestDefinition


@dataclass(frozen=True)
class NixlEPLaunch:
    """One concrete worker launch on a specific node."""

    node_idx: int
    num_processes: int
    include_tcp_server: bool
    append_output: bool = False


@dataclass(frozen=True)
class NixlEPStage:
    """Launches that should be appended when a given plan phase introduces new ranks."""

    idx: int
    launches: tuple[NixlEPLaunch, ...]


class NixlEPSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for the NIXL Elastic EP benchmark."""

    @property
    def tdef(self) -> NixlEPTestDefinition:
        return cast(NixlEPTestDefinition, self.test_run.test)

    def image_path(self) -> str | None:
        return str(self.tdef.docker_image.installed_path)

    def _container_mounts(self) -> list[str]:
        return []

    @property
    def num_processes_per_node(self) -> int:
        num_processes_per_node = self.tdef.cmd_args.num_processes_per_node
        if not isinstance(num_processes_per_node, int):
            raise ValueError("NIXL EP Slurm command generation requires num_processes_per_node to be an integer.")
        return num_processes_per_node

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
    def env_vars_path(self) -> Path:
        return self.test_run.output_path / "env_vars.sh"

    def node_log_path(self, node_idx: int) -> Path:
        return self.test_run.output_path / f"nixl-ep-node-{node_idx}.log"

    def resolve_plan_path(self) -> str:
        return str((self.test_run.output_path / GENERATED_PLAN_FILE_NAME).absolute())

    def _write_plan_file(self) -> None:
        plan_path = self.test_run.output_path / GENERATED_PLAN_FILE_NAME
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(self.tdef.cmd_args.parse_plan(), indent=2) + "\n", encoding="utf-8")

    @property
    def phase_transition_timeout_seconds(self) -> int:
        phase_timeout = 600
        if self.test_run.time_limit:
            phase_timeout = max(int(parse_time_limit(self.test_run.time_limit).total_seconds()), 1)
        num_plan_phases = max(len(self.tdef.cmd_args.parse_plan()), 1)
        return max(phase_timeout // num_plan_phases, 1)

    def _new_process_counts_by_phase(self) -> list[int]:
        counts: list[int] = []
        previous_positive_ranks: set[int] = set()
        for positive_ranks in [{rank for rank in phase if rank >= 0} for phase in self.tdef.cmd_args.parse_plan()]:
            counts.append(len(positive_ranks - previous_positive_ranks))
            previous_positive_ranks = positive_ranks

        self._validate_requested_processes(counts)
        return counts

    def _validate_requested_processes(self, new_process_counts: list[int]) -> None:
        total_requested_processes = sum(new_process_counts)
        num_nodes, _ = self.get_cached_nodes_spec()
        if num_nodes == 1:
            if self.num_processes_per_node != total_requested_processes:
                raise ValueError(
                    "For single-node NIXL EP runs, num_processes_per_node must match the plan-derived "
                    f"total launched workers ({total_requested_processes}), got {self.num_processes_per_node}."
                )
            return

        total_capacity = self.num_processes_per_node * num_nodes
        if total_requested_processes > total_capacity:
            raise ValueError(
                "For multi-node NIXL EP runs, num_processes_per_node defines the maximum number of workers "
                f"each node can launch across all plan phases. The plan requires {total_requested_processes} total "
                f"workers, but {num_nodes} nodes with capacity {self.num_processes_per_node} only provide "
                f"{total_capacity}."
            )

    def _allocate_stage_launches(
        self, phase_idx: int, new_process_count: int, remaining_capacity: list[int]
    ) -> tuple[NixlEPLaunch, ...]:
        if new_process_count == 0:
            return ()

        launches: list[NixlEPLaunch] = []
        remaining_phase_processes = new_process_count
        is_initial_phase = phase_idx == 0
        for node_idx, node_capacity in enumerate(remaining_capacity):
            if remaining_phase_processes == 0:
                break

            assignable = min(node_capacity, remaining_phase_processes)
            if assignable == 0:
                continue

            remaining_capacity[node_idx] -= assignable
            remaining_phase_processes -= assignable
            launches.append(
                NixlEPLaunch(
                    node_idx=node_idx,
                    num_processes=assignable,
                    include_tcp_server=(not is_initial_phase) or node_idx != 0,
                    append_output=not is_initial_phase,
                )
            )

        if remaining_phase_processes != 0:
            num_nodes, _ = self.get_cached_nodes_spec()
            raise ValueError(
                "For multi-node NIXL EP runs, the plan-derived launches cannot be packed onto "
                f"{num_nodes} nodes with per-node capacity {self.num_processes_per_node}. "
                f"Remaining phase size: {remaining_phase_processes}."
            )

        return tuple(launches)

    @property
    def plan_stages(self) -> tuple[NixlEPStage, ...]:
        new_process_counts = self._new_process_counts_by_phase()

        num_nodes, _ = self.get_cached_nodes_spec()
        remaining_capacity = [self.num_processes_per_node] * num_nodes
        stages: list[NixlEPStage] = []
        for phase_idx, new_process_count in enumerate(new_process_counts):
            launches = self._allocate_stage_launches(phase_idx, new_process_count, remaining_capacity)
            stages.append(NixlEPStage(idx=phase_idx, launches=launches))

        return tuple(stages)

    def _build_benchmark_command(self, launch: NixlEPLaunch) -> list[str]:
        cmd_args: NixlEPCmdArgs = self.tdef.cmd_args
        command = [
            cmd_args.python_executable,
            cmd_args.elastic_script,
            "--plan",
            self.resolve_plan_path(),
            "--num-processes",
            str(launch.num_processes),
        ]

        if launch.include_tcp_server:
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

    def _render_launch(self, launch: NixlEPLaunch) -> str:
        command = " ".join(self._build_benchmark_command(launch))
        env_file = self.env_vars_path.absolute()
        log_file = self.node_log_path(launch.node_idx).absolute()
        open_mode_arg = " --open-mode=append" if launch.append_output else ""
        script = f"source {shlex.quote(str(env_file))}; {command}".replace('"', '\\"')
        return f'{self._launch_srun_prefix(launch.node_idx)}{open_mode_arg} --output={log_file} bash -c "{script}"'

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
        with self.env_vars_path.open("w", encoding="utf-8") as env_file:
            for key, value in self.final_env_vars.items():
                env_file.write(f"export {key}={value}\n")

    def _background_launches_lines(self, launches: tuple[NixlEPLaunch, ...]) -> list[str]:
        lines: list[str] = []
        for launch in launches:
            lines.extend([self._render_launch(launch) + " &", "worker_pids+=($!)"])
        return lines

    @staticmethod
    def _finish_with_rc_lines() -> list[str]:
        return [
            'if [ "$rc" -eq 0 ]; then',
            '    echo "All NIXL EP launches completed successfully"',
            "fi",
            "",
            "exit $rc",
        ]

    @classmethod
    def _wait_for_workers_lines(cls) -> list[str]:
        return [
            "",
            "rc=0",
            'for pid in "${worker_pids[@]}"; do',
            '    wait "$pid" || rc=$?',
            "done",
            "",
            *cls._finish_with_rc_lines(),
        ]

    @staticmethod
    def _has_follower_launches(stages: list[NixlEPStage]) -> bool:
        return any(launch.node_idx != 0 for element in stages for launch in element.launches)

    def _render_single_stage(self, stage: NixlEPStage) -> str:
        return "\n".join(
            [
                'echo "Starting NIXL EP on the master node..."',
                self._render_launch(stage.launches[0]),
                "rc=$?",
                *self._finish_with_rc_lines(),
            ]
        )

    def _plan_helper_function_lines(self, has_followers: bool, has_multiple_stages: bool) -> list[str]:
        master_service_lines = [self.generate_wait_for_master_services_function(), ""] if has_followers else []
        phase_wait_lines = [self.generate_wait_for_phase_completion_function(), ""] if has_multiple_stages else []
        return [*master_service_lines, *phase_wait_lines]

    @staticmethod
    def _wait_for_master_services_lines() -> list[str]:
        return [
            "",
            'echo "Waiting for NIXL EP master services..."',
            "wait_for_master_services || exit 1",
        ]

    def _initial_follower_launch_lines(self, stage: NixlEPStage) -> list[str]:
        if len(stage.launches) == 1:
            return []
        return [
            "",
            f'echo "Starting the rest of initial phase {stage.idx}..."',
            *self._background_launches_lines(stage.launches[1:]),
        ]

    def _initial_stage_lines(self, stage: NixlEPStage, has_followers: bool) -> list[str]:
        primary_launch = stage.launches[0]
        master_service_lines = self._wait_for_master_services_lines() if has_followers else []
        header_lines = [
            "worker_pids=()",
            "",
            'echo "Starting initial NIXL EP stage on the master node..."',
            self._render_launch(primary_launch) + " &",
            "primary_pid=$!",
            "worker_pids+=($primary_pid)",
        ]
        return header_lines + master_service_lines + self._initial_follower_launch_lines(stage)

    def _followup_stage_lines(self, stage: NixlEPStage) -> list[str]:
        wait_phase = stage.idx - 1
        header_lines = [
            "",
            f'echo "Waiting for phase {wait_phase} before starting phase {stage.idx}..."',
            f'wait_for_phase_completion "{wait_phase}" "{self.node_log_path(0).absolute()}" "$primary_pid" || exit 1',
            "",
            f'echo "Starting launches for phase {stage.idx}..."',
        ]
        return header_lines + self._background_launches_lines(stage.launches)

    def _gen_srun_command(self) -> str:
        self._write_env_vars_file()
        self._write_plan_file()

        stages = [stage for stage in self.plan_stages if stage.launches]
        if not stages:
            raise ValueError("NIXL EP plan does not launch any non-negative ranks.")

        first_stage = stages[0]
        if len(stages) == 1 and len(first_stage.launches) == 1:
            return self._render_single_stage(first_stage)

        has_followers = self._has_follower_launches(stages)
        lines = self._plan_helper_function_lines(
            has_followers=has_followers,
            has_multiple_stages=len(stages) > 1,
        )
        lines += self._initial_stage_lines(first_stage, has_followers=has_followers)

        for stage in stages[1:]:
            lines += self._followup_stage_lines(stage)

        lines += self._wait_for_workers_lines()
        return "\n".join(lines)
