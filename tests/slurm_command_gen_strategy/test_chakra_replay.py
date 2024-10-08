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

import pytest
from cloudai.schema.test_template.chakra_replay.slurm_command_gen_strategy import ChakraReplaySlurmCommandGenStrategy
from cloudai.systems import SlurmSystem


class TestChakraReplaySlurmCommandGenStrategy:
    def get_slurm_args(
        self,
        slurm_system: SlurmSystem,
        job_name_prefix: str,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        num_nodes: int,
        nodes: List[str],
    ) -> Dict[str, Any]:
        return ChakraReplaySlurmCommandGenStrategy(slurm_system, {})._parse_slurm_args(
            job_name_prefix, env_vars, cmd_args, num_nodes, nodes
        )

    def get_test_command(self, slurm_system: SlurmSystem, cmd_args: Dict[str, str], extra_cmd_args: str) -> List[str]:
        return ChakraReplaySlurmCommandGenStrategy(slurm_system, {}).generate_test_command({}, cmd_args, extra_cmd_args)

    @pytest.mark.parametrize(
        "job_name_prefix, env_vars, cmd_args, num_nodes, nodes, expected_result",
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
        job_name_prefix: str,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        num_nodes: int,
        nodes: List[str],
        expected_result: Dict[str, Any],
        slurm_system: SlurmSystem,
    ) -> None:
        slurm_args = self.get_slurm_args(slurm_system, job_name_prefix, env_vars, cmd_args, num_nodes, nodes)
        assert slurm_args["image_path"] == expected_result["image_path"]
        assert slurm_args["container_mounts"] == expected_result["container_mounts"]

    def test_parse_slurm_args_invalid_cmd_args(self, slurm_system: SlurmSystem) -> None:
        job_name_prefix = "chakra_replay"
        env_vars = {"NCCL_DEBUG": "INFO"}
        cmd_args = {"trace_path": "/workspace/traces/"}  # Missing "docker_image_url"
        num_nodes = 2
        nodes = ["node1", "node2"]

        with pytest.raises(KeyError):
            self.get_slurm_args(slurm_system, job_name_prefix, env_vars, cmd_args, num_nodes, nodes)

    @pytest.mark.parametrize(
        "cmd_args, extra_cmd_args, expected_result",
        [
            (
                {"trace_type": "comms_trace", "trace_path": "/workspace/traces/", "backend": "nccl", "device": "gpu"},
                "--max-steps 100",
                [
                    "python /workspace/param/train/comms/pt/commsTraceReplay.py",
                    "--trace-type comms_trace",
                    "--trace-path /workspace/traces/",
                    "--backend nccl",
                    "--device gpu",
                    "--max-steps 100",
                ],
            ),
            (
                {"trace_type": "comms_trace", "trace_path": "/workspace/traces/", "backend": "nccl", "device": "gpu"},
                "",
                [
                    "python /workspace/param/train/comms/pt/commsTraceReplay.py",
                    "--trace-type comms_trace",
                    "--trace-path /workspace/traces/",
                    "--backend nccl",
                    "--device gpu",
                    "",
                ],
            ),
        ],
    )
    def test_generate_test_command(
        self, slurm_system: SlurmSystem, cmd_args: Dict[str, str], extra_cmd_args: str, expected_result: List[str]
    ) -> None:
        cmd = self.get_test_command(slurm_system, cmd_args, extra_cmd_args)
        assert cmd == expected_result

    def test_generate_test_command_invalid_args(self, slurm_system: SlurmSystem) -> None:
        cmd_args: Dict[str, str] = {"trace_type": "comms_trace", "backend": "nccl", "device": "gpu"}
        extra_cmd_args: str = "--max-steps 100"

        with pytest.raises(KeyError):
            self.get_test_command(slurm_system, cmd_args, extra_cmd_args)
