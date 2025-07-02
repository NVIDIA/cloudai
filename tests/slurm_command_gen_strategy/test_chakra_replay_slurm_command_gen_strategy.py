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

from typing import Any, Dict, List

import pytest

from cloudai import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.chakra_replay import ChakraReplayCmdArgs, ChakraReplaySlurmCommandGenStrategy
from tests.conftest import create_autospec_dataclass


class TestChakraReplaySlurmCommandGenStrategy:
    @pytest.mark.parametrize(
        "cmd_args, extra_cmd_args, expected_result",
        [
            (
                {"trace_type": "comms_trace", "trace_path": "/workspace/traces/", "num_replays": 10},
                "--max-steps 100",
                [
                    "comm_replay",
                    "--trace-type comms_trace",
                    "--trace-path /workspace/traces/",
                    "--num-replays 10",
                    "--max-steps 100",
                ],
            ),
            (
                {"trace_type": "comms_trace", "trace_path": "/workspace/traces/", "num_replays": 5},
                "",
                [
                    "comm_replay",
                    "--trace-type comms_trace",
                    "--trace-path /workspace/traces/",
                    "--num-replays 5",
                    "",
                ],
            ),
            (
                {
                    "trace_type": "comms_trace",
                    "trace_path": "/workspace/traces/",
                    "num_replays": 5,
                    "max_steps": "100",
                },
                "",
                [
                    "comm_replay",
                    "--trace-type comms_trace",
                    "--trace-path /workspace/traces/",
                    "--num-replays 5",
                    "--max-steps 100",
                    "",
                ],
            ),
        ],
    )
    def test_generate_test_command(
        self, cmd_args: Dict[str, Any], extra_cmd_args: str, expected_result: List[str], slurm_system: SlurmSystem
    ) -> None:
        tr = create_autospec_dataclass(TestRun)
        tr.test.test_definition.cmd_args = ChakraReplayCmdArgs(docker_image_url="", **cmd_args)
        tr.test.extra_cmd_args = extra_cmd_args
        cmd_gen_strategy = ChakraReplaySlurmCommandGenStrategy(slurm_system, tr)
        command = cmd_gen_strategy.generate_test_command({}, {})
        assert command == expected_result
