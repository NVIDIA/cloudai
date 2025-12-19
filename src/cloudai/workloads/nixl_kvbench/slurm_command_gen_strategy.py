# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path
from typing import cast

from cloudai.workloads.common.nixl import NIXLCmdGenBase

from .nixl_kvbench import NIXLKVBenchTestDefinition


class NIXLKVBenchSlurmCommandGenStrategy(NIXLCmdGenBase):
    """Command generation strategy for NIXLKVBench tests."""

    def _container_mounts(self) -> list[str]:
        mounts = []
        if filepath := self.tdef.cmd_args_dict.get("filepath"):
            local_dir = self.test_run.output_path / Path(f"{filepath}").name
            local_dir.mkdir(exist_ok=True)
            mounts.append(f"{local_dir.absolute()}:/{filepath}")
        return mounts

    @property
    def tdef(self) -> NIXLKVBenchTestDefinition:
        return cast(NIXLKVBenchTestDefinition, self.test_run.test)

    def image_path(self) -> str | None:
        return str(self.tdef.docker_image.installed_path)

    def _gen_srun_command(self) -> str:
        self._current_image_url = str(self.tdef.docker_image.installed_path)
        etcd_command: list[str] = self.gen_etcd_srun_command(self.tdef.cmd_args.etcd_path)
        kvbench_commands = self.gen_nixlbench_srun_commands(
            self.gen_kvbench_command(), str(self.tdef.cmd_args.backend or "unset")
        )
        self._current_image_url = None

        self.create_env_vars_file()

        final_cmd: list[str] = [
            " ".join(etcd_command),
            "etcd_pid=$!",
            " ".join(self.gen_wait_for_etcd_command(self.tdef.cmd_args.wait_etcd_for)),
            *[" ".join(cmd) + " &\nsleep 15" for cmd in kvbench_commands[:-1]],
            " ".join(kvbench_commands[-1]),
            " ".join(self.gen_kill_and_wait_cmd("etcd_pid")),
        ]
        return "\n".join(final_cmd)

    def gen_kvbench_command(self) -> list[str]:
        command: list[str] = [
            f"{self.tdef.cmd_args.python_executable}",
            f"{self.tdef.cmd_args.kvbench_script}",
            self.tdef.cmd_args.command,
        ]

        for k, v in self.test_run.test.cmd_args_dict.items():
            if v is None:
                continue

            key = "model_config" if k == "model_cfg" else k

            command.append(f"--{key} {v}")

        command.append("--etcd_endpoints http://$NIXL_ETCD_ENDPOINTS")

        return command
