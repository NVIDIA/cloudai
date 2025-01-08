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

from typing import Any, Dict, List, Union
from unittest.mock import Mock

import pytest

from cloudai import TestRun
from cloudai.schema.test_template.chakra_replay.slurm_command_gen_strategy import ChakraReplaySlurmCommandGenStrategy
from cloudai.systems import SlurmSystem
from tests.conftest import create_autospec_dataclass


class TestChakraReplaySlurmCommandGenStrategy:
    @pytest.fixture
    def cmd_gen_strategy(self, slurm_system: SlurmSystem) -> ChakraReplaySlurmCommandGenStrategy:
        return ChakraReplaySlurmCommandGenStrategy(slurm_system, {})

    @pytest.mark.parametrize(
        "job_name_prefix, env_vars, cmd_args_attrs, num_nodes, nodes, expected_result",
        [
            (
                "chakra_replay",
                {"NCCL_DEBUG": "INFO"},
                {"docker_image_url": "fake_image_url", "trace_path": "/workspace/traces/"},
                2,
                ["node1", "node2"],
                {
                    "image_path": "fake_image_url",
                    "container_mounts": "/workspace/traces/:/workspace/traces/",
                },
            ),
            (
                "chakra_replay",
                {"NCCL_DEBUG": "INFO"},
                {"docker_image_url": "another_image_url", "trace_path": "/another/trace_path/"},
                1,
                ["node1"],
                {
                    "image_path": "another_image_url",
                    "container_mounts": "/another/trace_path/:/another/trace_path/",
                },
            ),
        ],
    )
    def test_parse_slurm_args(
        self,
        cmd_gen_strategy: ChakraReplaySlurmCommandGenStrategy,
        job_name_prefix: str,
        env_vars: Dict[str, str],
        cmd_args_attrs: Dict[str, Any],
        num_nodes: int,
        nodes: List[str],
        expected_result: Dict[str, Any],
    ) -> None:
        mock_cmd_args = Mock(**cmd_args_attrs)
        mock_docker_image = Mock(installed_path=cmd_args_attrs["docker_image_url"])
        mock_test_definition = Mock(cmd_args=mock_cmd_args, docker_image=mock_docker_image)

        mock_test_run = Mock(
            num_nodes=num_nodes,
            nodes=nodes,
            test=Mock(test_definition=mock_test_definition),
        )

        slurm_args = cmd_gen_strategy._parse_slurm_args(job_name_prefix, env_vars, {}, mock_test_run)
        assert slurm_args["image_path"] == expected_result["image_path"]
        assert slurm_args["container_mounts"] == expected_result["container_mounts"]

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
        ],
    )
    def test_generate_test_command(
        self,
        cmd_gen_strategy: ChakraReplaySlurmCommandGenStrategy,
        cmd_args: Dict[str, Union[str, List[str]]],
        extra_cmd_args: str,
        expected_result: List[str],
        slurm_system: SlurmSystem,
    ) -> None:
        tr = create_autospec_dataclass(TestRun)
        tr.test.extra_cmd_args = extra_cmd_args
        command = cmd_gen_strategy.generate_test_command({}, cmd_args, tr)
        assert command == expected_result

    def test_generate_test_command_invalid_args(
        self, cmd_gen_strategy: ChakraReplaySlurmCommandGenStrategy, slurm_system: SlurmSystem
    ) -> None:
        cmd_args: Dict[str, Union[str, List[str]]] = {"trace_type": "comms_trace"}

        tr = create_autospec_dataclass(TestRun)
        tr.test.extra_cmd_args = "--max-steps 100"

        with pytest.raises(KeyError) as exc_info:
            cmd_gen_strategy.generate_test_command({}, cmd_args, tr)

        assert str(exc_info.value) == "'trace_path'", "Expected missing trace_path key"
