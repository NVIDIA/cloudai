# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import Mock

import pytest

from cloudai.core import Test, TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.ucc_test import UCCCmdArgs, UCCTestDefinition, UCCTestSlurmCommandGenStrategy


class TestUCCTestSlurmCommandGenStrategy:
    @pytest.fixture
    def cmd_gen_strategy(self, slurm_system: SlurmSystem) -> UCCTestSlurmCommandGenStrategy:
        return UCCTestSlurmCommandGenStrategy(slurm_system, {})

    @pytest.mark.parametrize(
        "cmd_args_data, extra_cmd_args, expected_command",
        [
            (
                {"collective": "allgather", "b": 8, "e": "256M", "docker_image_url": "url://fake/ucc"},
                {"--max-steps": "100"},
                [
                    "/opt/hpcx/ucc/bin/ucc_perftest",
                    "-c allgather",
                    "-b 8",
                    "-e 256M",
                    "-m cuda",
                    "-F",
                    "--max-steps=100",
                ],
            ),
            (
                {"collective": "allreduce", "b": 4, "e": "8M", "docker_image_url": "url://fake/ucc"},
                {},
                [
                    "/opt/hpcx/ucc/bin/ucc_perftest",
                    "-c allreduce",
                    "-b 4",
                    "-e 8M",
                    "-m cuda",
                    "-F",
                ],
            ),
        ],
    )
    def test_generate_test_command(
        self,
        tmp_path: Path,
        cmd_gen_strategy: UCCTestSlurmCommandGenStrategy,
        cmd_args_data: dict,
        extra_cmd_args: dict,
        expected_command: list[str],
    ) -> None:
        ucc_cmd_args = UCCCmdArgs(
            docker_image_url=cmd_args_data["docker_image_url"],
            collective=cmd_args_data["collective"],
            b=cmd_args_data["b"],
            e=cmd_args_data.get("e", "8M"),
        )

        test_def = UCCTestDefinition(
            name="ucc_test",
            description="UCC test",
            test_template_name="default_template",
            cmd_args=ucc_cmd_args,
            extra_env_vars={},
            extra_cmd_args=extra_cmd_args,
        )

        test_obj = Test(test_definition=test_def, test_template=Mock())

        tr = TestRun(
            test=test_obj,
            num_nodes=1,
            nodes=[],
            output_path=tmp_path / "output",
            name="test-job",
        )

        command = cmd_gen_strategy.generate_test_command(
            test_def.extra_env_vars,
            test_def.cmd_args_dict,
            tr,
        )
        assert command == expected_command
