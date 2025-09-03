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

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmCommandGenStrategy, SlurmSystem

from .nixl_kvbench import NIXLKVBenchTestDefinition


class NIXLKVBenchSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for NIXLKVBench tests."""

    def __init__(self, system: SlurmSystem, test_run: TestRun) -> None:
        super().__init__(system, test_run)

    def _container_mounts(self) -> list[str]:
        return []

    @property
    def tdef(self) -> NIXLKVBenchTestDefinition:
        return cast(NIXLKVBenchTestDefinition, self.test_run.test.test_definition)

    def generate_test_command(self) -> list[str]:
        return self.gen_kvbench_command()

    def gen_kvbench_command(self) -> list[str]:
        command: list[str] = [f"{self.tdef.cmd_args.python_executable}", f"{self.tdef.cmd_args.kvbench_script}"]
        for k, v in self.test_run.test.test_definition.cmd_args_dict.items():
            command.append(f"--{k} {v}")

        if self.tdef.cmd_args.with_etcd:
            command.append("--etcd-endpoints http://$SERVER:2379")

        return command
