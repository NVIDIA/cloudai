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

from cloudai.workloads.sglang import SglangArgs, SglangCmdArgs, SglangTestDefinition


@pytest.mark.parametrize(
    "prefill_gpu_ids, decode_gpu_ids",
    [("0,1", "0,1"), (None, None), (None, "11,42")],
)
def test_valid_gpu_ids_configuration(prefill_gpu_ids: str | None, decode_gpu_ids: str | None) -> None:
    prefill = None
    if prefill_gpu_ids is not None:
        prefill = SglangArgs(gpu_ids=prefill_gpu_ids)

    decode = SglangArgs(gpu_ids=decode_gpu_ids)
    tdef = SglangTestDefinition(
        name="test",
        description="test",
        test_template_name="sglang",
        cmd_args=SglangCmdArgs(docker_image_url="test_url", prefill=prefill, decode=decode),
    )

    if prefill_gpu_ids is not None:
        assert tdef.cmd_args.prefill
        assert tdef.cmd_args.prefill.gpu_ids == prefill_gpu_ids

    assert tdef.cmd_args.decode.gpu_ids == decode_gpu_ids


@pytest.mark.parametrize(
    "prefill_gpu_ids, decode_gpu_ids",
    [("0,1", None), (None, "0,1")],
)
def test_invalid_gpu_ids_configuration(prefill_gpu_ids: str | None, decode_gpu_ids: str | None) -> None:
    prefill = SglangArgs(gpu_ids=prefill_gpu_ids)
    decode = SglangArgs(gpu_ids=decode_gpu_ids)
    with pytest.raises(ValueError):
        SglangTestDefinition(
            name="test",
            description="test",
            test_template_name="sglang",
            cmd_args=SglangCmdArgs(docker_image_url="test_url", prefill=prefill, decode=decode),
        )
