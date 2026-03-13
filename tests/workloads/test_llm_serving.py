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

import pytest

from cloudai.workloads.common.llm_serving import all_gpu_ids
from cloudai.workloads.sglang import SglangArgs, SglangCmdArgs, SglangTestDefinition
from cloudai.workloads.vllm import VllmArgs, VllmCmdArgs, VllmTestDefinition

LLMTestDefinition = VllmTestDefinition | SglangTestDefinition


@pytest.fixture(params=["vllm", "sglang"])
def llm_tdef(request: pytest.FixtureRequest) -> LLMTestDefinition:
    if request.param == "vllm":
        return VllmTestDefinition(
            name="vllm_test",
            description="vLLM benchmark test",
            test_template_name="Vllm",
            cmd_args=VllmCmdArgs(docker_image_url="nvcr.io/nvidia/vllm:latest"),
        )
    return SglangTestDefinition(
        name="sglang_test",
        description="SGLang benchmark test",
        test_template_name="sglang",
        cmd_args=SglangCmdArgs(docker_image_url="docker.io/lmsysorg/sglang:dev"),
    )


class TestAllGpuIds:
    @pytest.mark.parametrize("cuda_visible_devices", ["0", "0,1,2,3", "0,1,2,3,4,5,6,7"])
    def test_from_cuda_visible_devices(self, llm_tdef: LLMTestDefinition, cuda_visible_devices: str) -> None:
        llm_tdef.extra_env_vars = {"CUDA_VISIBLE_DEVICES": cuda_visible_devices}

        assert all_gpu_ids(llm_tdef, 8) == [int(gpu_id) for gpu_id in cuda_visible_devices.split(",")]

    @pytest.mark.parametrize("gpus_per_node", [None, 1, 8])
    def test_fallback_to_system_gpu_count(self, llm_tdef: LLMTestDefinition, gpus_per_node: int | None) -> None:
        llm_tdef.extra_env_vars = {}

        assert all_gpu_ids(llm_tdef, gpus_per_node) == list(range(gpus_per_node or 1))

    def test_prefill_and_decode_gpu_ids_override_cuda_visible_devices(self, llm_tdef: LLMTestDefinition) -> None:
        llm_tdef.extra_env_vars = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
        if isinstance(llm_tdef, VllmTestDefinition):
            llm_tdef.cmd_args.prefill = VllmArgs(gpu_ids="4")
            llm_tdef.cmd_args.decode.gpu_ids = "5"
        else:
            llm_tdef.cmd_args.prefill = SglangArgs(gpu_ids="4")
            llm_tdef.cmd_args.decode.gpu_ids = "5"

        assert all_gpu_ids(llm_tdef, 4) == [4, 5]
