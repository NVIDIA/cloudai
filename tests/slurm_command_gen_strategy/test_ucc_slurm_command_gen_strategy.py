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

from typing import Dict, List, Literal, Union, cast
from unittest.mock import Mock

import pytest

from cloudai.systems import SlurmSystem
from cloudai.workloads.ucc_test import UCCCmdArgs, UCCTestDefinition, UCCTestSlurmCommandGenStrategy


class TestUCCTestSlurmCommandGenStrategy:
    @pytest.fixture
    def cmd_gen_strategy(self, slurm_system: SlurmSystem) -> UCCTestSlurmCommandGenStrategy:
        return UCCTestSlurmCommandGenStrategy(slurm_system, {})

    @pytest.mark.parametrize(
        "cmd_args, extra_cmd_args, expected_command",
        [
            (
                {"collective": "allgather", "b": 8, "e": "256M"},
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
                {"collective": "allreduce", "b": 4, "e": "8M"},
                "",
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
        cmd_gen_strategy: UCCTestSlurmCommandGenStrategy,
        cmd_args: Dict[str, Union[str, List[str]]],
        extra_cmd_args: str,
        expected_command: List[str],
    ) -> None:
        env_vars = {}
        tr = Mock()
        tr.test.extra_cmd_args = extra_cmd_args

        mock_cmd_args = UCCCmdArgs(
            collective=cast(Union[Literal["allgather"], Literal["allreduce"]], cmd_args["collective"]),
            b=cast(int, cmd_args["b"]),
            e=cmd_args.get("e", "8M"),
        )
        mock_test_definition = Mock(spec=UCCTestDefinition)
        mock_test_definition.cmd_args = mock_cmd_args
        tr.test.test_definition = mock_test_definition

        command = cmd_gen_strategy.generate_test_command(env_vars, {}, tr)
        assert command == expected_command
