# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Dict, List

import pytest
from cloudai.schema.test_template.ucc_test.slurm_command_gen_strategy import UCCTestSlurmCommandGenStrategy
from cloudai.systems import SlurmSystem


class TestUCCTestSlurmCommandGenStrategy:
    def get_test_command(
        self, slurm_system: SlurmSystem, env_vars: Dict[str, str], cmd_args: Dict[str, str], extra_cmd_args: str
    ) -> List[str]:
        return UCCTestSlurmCommandGenStrategy(slurm_system, {}).generate_test_command(
            env_vars, cmd_args, extra_cmd_args
        )

    @pytest.mark.parametrize(
        "env_vars, cmd_args, extra_cmd_args, expected_command",
        [
            (
                {},
                {"collective": "allgather", "b": "8", "e": "256M"},
                "--max-steps 100",
                [
                    "/opt/hpcx/ucc/bin/ucc_perftest",
                    "-c allgather",
                    "-b 8",
                    "-e 256M",
                    "-m cuda",
                    "-F",
                    "--max-steps 100",
                ],
            ),
            (
                {},
                {"collective": "allreduce", "b": "4"},
                "",
                [
                    "/opt/hpcx/ucc/bin/ucc_perftest",
                    "-c allreduce",
                    "-b 4",
                    "-m cuda",
                    "-F",
                ],
            ),
        ],
    )
    def test_generate_test_command(
        self,
        slurm_system: SlurmSystem,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_cmd_args: str,
        expected_command: List[str],
    ) -> None:
        command = self.get_test_command(slurm_system, env_vars, cmd_args, extra_cmd_args)
        assert command == expected_command
