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

from typing import cast

from cloudai.core import CmdArgs, Installable, TestDefinition
from cloudai.systems.slurm import SlurmCommandGenStrategy


class BashCmdArgs(CmdArgs):
    """Arguments for a Bash command."""

    cmd: str


class BashCmdTestDefinition(TestDefinition):
    """Test definition for a Bash command."""

    cmd_args: BashCmdArgs

    @property
    def installables(self) -> list[Installable]:
        return [*self.git_repos]


class BashCmdCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for generic Slurm container tests."""

    def _container_mounts(self) -> list[str]:
        return []

    def gen_nsys_command(self) -> list[str]:
        """NSYS command is generated as part of the test command and disabled here."""
        return []

    def gen_srun_prefix(self, use_pretest_extras: bool = False) -> list[str]:  # noqa: Vulture
        return []

    def generate_test_command(self) -> list[str]:
        tdef: BashCmdTestDefinition = cast(BashCmdTestDefinition, self.test_run.test)
        srun_command_parts: list[str] = [*super().gen_nsys_command(), tdef.cmd_args.cmd]
        return [" ".join(srun_command_parts)]

    def gen_srun_success_check(self) -> str:
        return "[ $? -eq 0 ] && echo 1 || echo 0"
