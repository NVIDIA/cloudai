# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.vllm import VllmCmdArgs, VllmSlurmCommandGenStrategy, VllmTestDefinition


@pytest.fixture
def vllm() -> VllmTestDefinition:
    return VllmTestDefinition(
        name="vllm_test",
        description="vLLM benchmark test",
        test_template_name="Vllm",
        cmd_args=VllmCmdArgs(docker_image_url="nvcr.io/nvidia/vllm:latest", model="Qwen/Qwen3-0.6B", port=8000),
    )


@pytest.fixture
def vllm_tr(vllm: VllmTestDefinition, tmp_path: Path) -> TestRun:
    return TestRun(test=vllm, num_nodes=1, nodes=[], output_path=tmp_path, name="vllm-job")


class TestVllmSlurmCommandGenStrategy:
    """Test the VllmSlurmCommandGenStrategy class."""

    def test_generate_vllm_command(self, vllm_tr: TestRun, slurm_system: SlurmSystem) -> None:
        cmd_gen_strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)

        command = " ".join(cmd_gen_strategy.get_vllm_serve_command())

        assert command == f"vllm serve {vllm_tr.test.cmd_args.model} --port {vllm_tr.test.cmd_args.port}"
