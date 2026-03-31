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

from cloudai.core import GitRepo, TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.vllm import VllmArgs, VllmCmdArgs, VllmTestDefinition


def test_vllm_serve_args_exclude_internal_fields() -> None:
    assert VllmArgs(gpu_ids="0", nixl_threads=1).serve_args == []


def test_vllm_serve_args_convert_boolean_flags() -> None:
    assert VllmArgs.model_validate({"enable_expert_parallel": True}).serve_args == ["--enable-expert-parallel"]
    assert VllmArgs.model_validate({"enable_expert_parallel": False}).serve_args == ["--no-enable-expert-parallel"]
    assert VllmArgs.model_validate({"tokens_only": True}).serve_args == ["--tokens-only"]
    assert VllmArgs.model_validate({"tokens_only": False}).serve_args == ["--no-tokens-only"]


def test_vllm_serve_args_convert_standalone_boolean_flags() -> None:
    assert VllmArgs.model_validate({"headless": True}).serve_args == ["--headless"]
    assert VllmArgs.model_validate({"headless": False}).serve_args == ["--no-headless"]


def test_vllm_serve_args_keep_non_boolean_values() -> None:
    assert VllmArgs.model_validate({"tensor_parallel_size": 4}).serve_args == ["--tensor-parallel-size", "4"]


def test_installables_include_proxy_script_repo() -> None:
    proxy_script_repo = GitRepo(url="./proxy_script_repo", commit="commit")
    tdef = VllmTestDefinition(
        name="test",
        description="test",
        test_template_name="vllm",
        cmd_args=VllmCmdArgs(docker_image_url="test_url"),
        proxy_script_repo=proxy_script_repo,
    )

    assert tdef.installables == [tdef.docker_image, tdef.hf_model, proxy_script_repo]


def test_constraint_check_rejects_tp_pp_dp_above_available_gpus(tmp_path) -> None:
    tdef = VllmTestDefinition(
        name="test",
        description="test",
        test_template_name="vllm",
        cmd_args=VllmCmdArgs(
            docker_image_url="test_url",
            decode=VllmArgs.model_validate({"tensor_parallel_size": 2, "pipeline_parallel_size": 2}),
        ),
        extra_env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2"},
    )
    tr = TestRun(name="vllm", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)

    assert tdef.constraint_check(tr, None) is False


def test_constraint_check_rejects_flashinfer_with_dp_and_expert_parallel(tmp_path) -> None:
    tdef = VllmTestDefinition(
        name="test",
        description="test",
        test_template_name="vllm",
        cmd_args=VllmCmdArgs(
            docker_image_url="test_url",
            decode=VllmArgs.model_validate(
                {"data_parallel_size": 2, "all2all_backend": "flashinfer_all2allv", "enable_expert_parallel": True}
            ),
        ),
        extra_env_vars={"CUDA_VISIBLE_DEVICES": "0,1"},
    )
    tr = TestRun(name="vllm", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)

    assert tdef.constraint_check(tr, None) is False


def test_constraint_check_validates_disaggregated_roles_against_split_gpus(tmp_path) -> None:
    tdef = VllmTestDefinition(
        name="test",
        description="test",
        test_template_name="vllm",
        cmd_args=VllmCmdArgs(
            docker_image_url="test_url",
            prefill=VllmArgs.model_validate({"tensor_parallel_size": 3}),
            decode=VllmArgs.model_validate({"tensor_parallel_size": 2}),
        ),
        extra_env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    )
    tr = TestRun(name="vllm", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)

    assert tdef.constraint_check(tr, None) is False


def test_constraint_check_uses_all_node_gpus_per_role_for_two_node_disagg(tmp_path, slurm_system: SlurmSystem) -> None:
    tdef = VllmTestDefinition(
        name="test",
        description="test",
        test_template_name="vllm",
        cmd_args=VllmCmdArgs(
            docker_image_url="test_url",
            prefill=VllmArgs.model_validate({"tensor_parallel_size": 4}),
            decode=VllmArgs.model_validate({"tensor_parallel_size": 4}),
        ),
    )
    tr = TestRun(name="vllm", test=tdef, num_nodes=2, nodes=[], output_path=tmp_path)
    slurm_system.gpus_per_node = 4

    assert tdef.constraint_check(tr, slurm_system) is True
