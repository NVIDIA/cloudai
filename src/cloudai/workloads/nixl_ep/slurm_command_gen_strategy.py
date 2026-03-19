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
from pathlib import Path
from typing import List, cast

from cloudai.systems.slurm import SlurmCommandGenStrategy
from cloudai.util import parse_time_limit

from .nixl_ep import NixlEPCmdArgs, NixlEPTestDefinition


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
    def final_env_vars(self) -> dict[str, str | list[str]]:
        env_vars = super().final_env_vars

        plugin_dir = str(env_vars.get("NIXL_PLUGIN_DIR", "")).strip()
        if not plugin_dir or plugin_dir.startswith(f"{self.tdef.container_runtime_root}/"):
            env_vars["NIXL_PLUGIN_DIR"] = self.tdef.container_plugin_dir

        python_path = str(env_vars.get("PYTHONPATH", "")).strip()
        if not python_path:
            env_vars["PYTHONPATH"] = self.tdef.container_python_path
        elif self.tdef.container_python_path not in python_path.split(":"):
            env_vars["PYTHONPATH"] = f"{self.tdef.container_python_path}:{python_path}"

        library_dir = self.tdef.container_library_dir
        ld_library_path = str(env_vars.get("LD_LIBRARY_PATH", "")).strip()
        if not ld_library_path:
            env_vars["LD_LIBRARY_PATH"] = f"{library_dir}:$LD_LIBRARY_PATH"
        elif library_dir not in ld_library_path.split(":"):
            env_vars["LD_LIBRARY_PATH"] = f"{library_dir}:{ld_library_path}"

        if self.tdef.cmd_args.debug_logging:
            env_vars.setdefault("PYTHONUNBUFFERED", "1")
            env_vars.setdefault("NIXL_DEBUG_LOGGING", "yes")
            env_vars.setdefault("NIXL_LOG_LEVEL", self.tdef.cmd_args.nixl_log_level or "TRACE")
            env_vars.setdefault("UCX_LOG_LEVEL", self.tdef.cmd_args.ucx_log_level or "DEBUG")
            env_vars.setdefault("TORCH_CPP_LOG_LEVEL", "INFO")
            env_vars.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")

        return env_vars

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

    def _append_sbatch_directives(self, batch_script_content: List[str]) -> None:
        super()._append_sbatch_directives(batch_script_content)
        batch_script_content.extend(
            [
                "",
                "nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )",
                "nodes_array=($nodes)",
                "master_node=${nodes_array[0]}",
                'master_ip=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address | awk \'{print $1}\')',
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
            json.dumps(self.tdef.cmd_args.parse_plan(), indent=2) + "\n",
            encoding="utf-8",
        )
        return str(self.generated_plan_path.absolute())

    @property
    def inline_plan(self) -> list[list[int]] | None:
        if self.tdef.cmd_args.plan is None:
            return None
        return self.tdef.cmd_args.parse_plan()

    @property
    def phase_transition_timeout_seconds(self) -> int:
        if self.test_run.time_limit:
            return max(int(parse_time_limit(self.test_run.time_limit).total_seconds()), 1)
        return 600

    @property
    def single_node_launch_waves(self) -> list[tuple[int | None, int]]:
        raw = self.tdef.cmd_args.num_processes_per_node
        if not isinstance(raw, int) or self.test_run.num_nodes != 1:
            return []

        plan = self.inline_plan
        if not plan:
            return [(None, raw)]

        positive_phases = [{rank for rank in phase if rank >= 0} for phase in plan]
        waves: list[tuple[int | None, int]] = [(None, len(positive_phases[0]))]
        for phase_idx in range(1, len(positive_phases)):
            added_positive = positive_phases[phase_idx] - positive_phases[phase_idx - 1]
            if added_positive:
                waves.append((phase_idx - 1, len(added_positive)))

        expected_total_processes = sum(num_processes for _, num_processes in waves)
        if raw != expected_total_processes:
            raise ValueError(
                "For single-node NIXL EP runs, num_processes_per_node must match the plan-derived "
                f"launch waves total ({expected_total_processes}), got {raw}."
            )

        return waves

    def build_elastic_command(self, num_processes: int, include_tcp_server: bool = False) -> list[str]:
        cmd_args: NixlEPCmdArgs = self.tdef.cmd_args
        command = [
            cmd_args.python_executable,
            self.tdef.resolve_elastic_script_path(),
            "--plan",
            self.resolve_plan_path(),
            "--num-processes",
            str(num_processes),
            "--num-tokens",
            str(cmd_args.num_tokens),
            "--num-experts-per-rank",
            str(cmd_args.num_experts_per_rank),
            "--hidden-dim",
            str(cmd_args.hidden_dim),
            "--num-topk",
            str(cmd_args.num_topk),
        ]

        if include_tcp_server:
            command.extend(["--tcp-server", "$master_ip"])
        if cmd_args.disable_ll_nvlink:
            command.append("--disable-ll-nvlink")
        if cmd_args.kineto:
            command.append("--kineto")

        for arg, value in sorted((cmd_args.model_extra or {}).items()):
            flag = "--" + arg.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            elif value is not None:
                command.extend([flag, str(value)])

        return command

    def generate_wait_for_master_services_function(self) -> str:
        cmd_args = self.tdef.cmd_args
        return f"""\
wait_for_master_services() {{
    local timeout={cmd_args.service_startup_timeout_seconds}
    local interval=1
    local end_time=$(($(date +%s) + timeout))

    while [ "$(date +%s)" -lt "$end_time" ]; do
        if timeout 1 bash -c ": > /dev/tcp/$master_ip/{cmd_args.store_port}" >/dev/null 2>&1 && \\
           timeout 1 bash -c ": > /dev/tcp/$master_ip/{cmd_args.rank_server_port}" >/dev/null 2>&1; then
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
        script = self._launch_script(node_idx, env_file, command).replace('"', '\\"')
        return (
            f'{self._launch_srun_prefix(node_idx)}{open_mode_arg} --output={log_file} '
            f'bash -c "{script}"'
        )

    def _launch_script(self, node_idx: int, env_file: Path, command: str) -> str:
        source_env = f"source {shlex.quote(str(env_file))}"
        if not self.tdef.cmd_args.debug_logging:
            return f"{source_env}; {command}"

        return "; ".join([source_env, self._debug_diagnostics_command(node_idx), "set -x", command])

    def _debug_diagnostics_command(self, node_idx: int) -> str:
        marker_file = shlex.quote(str((self.test_run.output_path / f"nixl-ep-node-{node_idx}.debug.once").absolute()))
        plan_file = shlex.quote(str(self.generated_plan_path.absolute()))
        commands = [
            "  echo '=== NIXL EP debug diagnostics start ==='",
            "  date",
            "  hostname",
            "  pwd",
            "  if command -v python3 >/dev/null 2>&1; then python3 -c 'import os; prefixes=(\"CUDA\",\"UCX\",\"NIXL\",\"NCCL\",\"TORCH\"); keep={\"LD_LIBRARY_PATH\",\"PYTHONPATH\",\"NIXL_PLUGIN_DIR\",\"PYTHONUNBUFFERED\"}; [print(f\"{k}={os.environ[k]}\") for k in sorted(os.environ) if k.startswith(prefixes) or k in keep]'; fi",
            "  if command -v python3 >/dev/null 2>&1; then python3 --version; fi",
            "  if command -v python3 >/dev/null 2>&1; then python3 -c 'import nixl_ep, torch; print(\"nixl_ep=\", nixl_ep.__file__); print(\"torch=\", torch.__version__)'; fi",
            "  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L; fi",
            "  if command -v rdma >/dev/null 2>&1; then rdma link show; fi",
            "  if [ -e /dev/infiniband ]; then ls -al /dev/infiniband; else echo '/dev/infiniband not present'; fi",
            "  if command -v ibv_devinfo >/dev/null 2>&1; then ibv_devinfo -l; fi",
            "  if command -v ucx_info >/dev/null 2>&1; then ucx_info -d; fi",
            f"  echo '--- Generated NIXL EP plan ({self.GENERATED_PLAN_FILE_NAME}) ---'",
            f"  if [ -f {plan_file} ]; then cat {plan_file}; fi",
            f"  touch {marker_file}",
            "  echo '=== NIXL EP debug diagnostics end ==='",
        ]
        body = "; ".join(command.strip() for command in commands)
        return f"if [ ! -f {marker_file} ]; then {body}; fi"

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

    def _gen_srun_command(self) -> str:
        self._write_env_vars_file()
        processes_per_node = self.processes_per_node

        if len(processes_per_node) == 1:
            waves = self.single_node_launch_waves
            if len(waves) <= 1:
                single_wave_processes = waves[0][1] if waves else processes_per_node[0]
                return "\n".join(
                    [
                        'echo "Starting NIXL EP on the master node..."',
                        self._launch_srun_command(0, single_wave_processes),
                    ]
                )

            primary_log_file = (self.test_run.output_path / "nixl-ep-node-0.log").absolute()
            lines = [
                self.generate_wait_for_phase_completion_function(),
                "",
                "worker_pids=()",
                "",
                'echo "Starting initial NIXL EP wave on the master node..."',
                self._launch_srun_command(0, waves[0][1]) + " &",
                "primary_pid=$!",
                "worker_pids+=($primary_pid)",
            ]

            for wave_idx, (trigger_phase, num_processes) in enumerate(waves[1:], start=1):
                if trigger_phase is None:
                    raise ValueError("Only the first single-node NIXL EP launch wave may omit a trigger phase.")
                lines.extend(
                    [
                        "",
                        f'echo "Waiting for phase {trigger_phase} before starting wave {wave_idx}..."',
                        f'wait_for_phase_completion "{trigger_phase}" "{primary_log_file}" "$primary_pid" || exit 1',
                        f'echo "Starting NIXL EP wave {wave_idx} on the master node..."',
                        self._launch_srun_command(
                            0,
                            num_processes,
                            include_tcp_server=True,
                            append_output=True,
                        )
                        + " &",
                        "worker_pids+=($!)",
                    ]
                )

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
            return "\n".join(lines)

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
        return "\n".join(lines)
