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

from typing import Any, Dict, List
from unittest.mock import Mock

import pytest
from cloudai._core.test_scenario import TestRun
from cloudai.schema.test_template.nccl_test.slurm_command_gen_strategy import NcclTestSlurmCommandGenStrategy
from cloudai.systems import SlurmSystem


class TestNcclTestSlurmCommandGenStrategy:
    @pytest.fixture
    def cmd_gen_strategy(self, slurm_system: SlurmSystem) -> NcclTestSlurmCommandGenStrategy:
        return NcclTestSlurmCommandGenStrategy(slurm_system, {})

    @pytest.mark.parametrize(
        "job_name_prefix, env_vars, cmd_args, num_nodes, nodes, expected_result",
        [
            (
                "nccl_test",
                {"NCCL_TOPO_FILE": "/path/to/topo", "DOCKER_NCCL_TOPO_FILE": "/docker/topo"},
                {"subtest_name": "all_reduce_perf", "docker_image_url": "fake_image_url"},
                2,
                ["node1", "node2"],
                {
                    "container_mounts": "/path/to/topo:/docker/topo",
                },
            ),
            (
                "nccl_test",
                {"NCCL_TOPO_FILE": "/path/to/topo"},
                {"subtest_name": "all_reduce_perf", "docker_image_url": "another_image_url"},
                1,
                ["node1"],
                {
                    "container_mounts": "",
                },
            ),
        ],
    )
    def test_parse_slurm_args(
        self,
        cmd_gen_strategy: NcclTestSlurmCommandGenStrategy,
        job_name_prefix: str,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        num_nodes: int,
        nodes: List[str],
        expected_result: Dict[str, Any],
    ) -> None:
        tr = Mock(spec=TestRun)
        tr.nodes = nodes
        tr.num_nodes = num_nodes
        slurm_args = cmd_gen_strategy._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)
        assert slurm_args["container_mounts"] == expected_result["container_mounts"]

    @pytest.mark.parametrize(
        "cmd_args, extra_cmd_args, expected_command",
        [
            (
                {"subtest_name": "all_reduce_perf", "nthreads": "4", "ngpus": "2"},
                "--max-steps 100",
                [
                    "/usr/local/bin/all_reduce_perf",
                    "--nthreads 4",
                    "--ngpus 2",
                    "--max-steps 100",
                ],
            ),
            (
                {"subtest_name": "all_reduce_perf", "op": "sum", "datatype": "float"},
                "",
                [
                    "/usr/local/bin/all_reduce_perf",
                    "--op sum",
                    "--datatype float",
                ],
            ),
        ],
    )
    def test_generate_test_command(
        self,
        cmd_gen_strategy: NcclTestSlurmCommandGenStrategy,
        cmd_args: Dict[str, str],
        extra_cmd_args: str,
        expected_command: List[str],
    ) -> None:
        env_vars = {}
        command = cmd_gen_strategy.generate_test_command(env_vars, cmd_args, extra_cmd_args)
        assert command == expected_command
