# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import cast

from cloudai.workloads.common.nixl import NIXLCmdGenBase

from .nixl_bench import NIXLBenchTestDefinition

ASIO_PROCESS_START_DELAY_SECONDS = 4
ETCD_PROCESS_START_DELAY_SECONDS = 15


class NIXLBenchSlurmCommandGenStrategy(NIXLCmdGenBase):
    """Command generation strategy for NIXL Bench tests."""

    @property
    def tdef(self) -> NIXLBenchTestDefinition:
        return cast(NIXLBenchTestDefinition, self.test_run.test)

    def _gen_srun_command(self) -> str:
        self.create_env_vars_file()

        backend = str(self.tdef.cmd_args_dict.get("backend", "unset"))
        self._current_image_url = str(self.tdef.docker_image.installed_path)
        try:
            nixl_commands = self.gen_nixlbench_srun_commands(self.gen_nixlbench_command(), backend)
            if self.tdef.cmd_args.runtime_type == "ASIO" and len(nixl_commands) != 2:
                raise ValueError(f"ASIO runtime requires exactly two NIXLBench processes, got {len(nixl_commands)}.")

            process_start_delay = (
                ASIO_PROCESS_START_DELAY_SECONDS
                if self.tdef.cmd_args.runtime_type == "ASIO"
                else ETCD_PROCESS_START_DELAY_SECONDS
            )
            commands = ["nixl_pids=()"]
            for cmd in nixl_commands[:-1]:
                commands.extend(
                    [
                        " ".join(cmd) + " &",
                        'nixl_pids+=("$!")',
                        f"sleep {process_start_delay}",
                    ]
                )

            commands.extend(
                [
                    " ".join(nixl_commands[-1]),
                    "nixl_rc=$?",
                    'for nixl_pid in "${nixl_pids[@]}"; do',
                    '  wait "$nixl_pid"',
                    "  wait_rc=$?",
                    '  if [ "$nixl_rc" -eq 0 ] && [ "$wait_rc" -ne 0 ]; then',
                    "    nixl_rc=$wait_rc",
                    "  fi",
                    "done",
                ]
            )
            if not self.tdef.uses_etcd:
                return "\n".join([*commands, '(exit "$nixl_rc")'])

            etcd_command: list[str] = self.gen_etcd_srun_command(self.tdef.cmd_args.etcd_path)
        finally:
            self._current_image_url = None

        commands = [
            " ".join(etcd_command),
            "etcd_pid=$!",
            " ".join(self.gen_wait_for_etcd_command(self.tdef.cmd_args.wait_etcd_for)),
            *commands,
            " ".join(self.gen_kill_and_wait_cmd("etcd_pid")),
            '(exit "$nixl_rc")',
        ]
        return "\n".join(commands)

    def gen_nixlbench_command(self) -> list[str]:
        tdef: NIXLBenchTestDefinition = cast(NIXLBenchTestDefinition, self.test_run.test)
        cmd = [tdef.cmd_args.path_to_benchmark]

        for k, v in tdef.cmd_args_dict.items():
            if k == "etcd_endpoints":
                k = "etcd-endpoints"
            cmd.append(f"--{k}={v}")

        return cmd
