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

from cloudai.core import GitRepo
from cloudai.workloads.vllm import VllmArgs, VllmCmdArgs, VllmTestDefinition


def test_vllm_serve_args_exclude_internal_fields() -> None:
    assert VllmArgs(gpu_ids="0", nixl_threads=1).serve_args == []


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
