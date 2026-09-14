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

from cloudai.core import GitRepo, TestRun
from cloudai.workloads.megatron_bridge import MegatronBridgeCmdArgs, MegatronBridgeTestDefinition


def _megatron_bridge_tdef() -> MegatronBridgeTestDefinition:
    """
    Build a test definition whose fixed arguments satisfy every parallelism constraint.

    num_gpus=128 with tp=cp=et=1 and mb=1, gb=128 keeps num_gpus % (tp*pp*cp), num_gpus % (et*ep*pp) and
    gb % (mb*dp) at zero for every (ep, pp) pair exercised below, so each case isolates the pipelining rules.
    """
    return MegatronBridgeTestDefinition(
        name="mb",
        description="desc",
        test_template_name="MegatronBridge",
        cmd_args=MegatronBridgeCmdArgs(
            hf_token="dummy_token",
            model_family_name="qwen3",
            model_recipe_name="30b_a3b",
            num_gpus=128,
            tp=1,
            cp=1,
            et=1,
            mb=1,
            gb=128,
        ),
        git_repos=[GitRepo(url="https://github.com/NVIDIA-NeMo/Megatron-Bridge.git", commit="r0.2.0")],
    )


@pytest.mark.parametrize(
    "ep,pp,vp,a2a,res",
    [
        (32, 1, None, False, True),  # no pipeline and no virtual pipeline
        (32, 1, 1, False, True),  # vp=1 is off, command gen rewrites it to -vp None
        (32, 1, 2, False, False),  # vp > 1 needs pp > 1
        (32, 1, 4, False, False),  # vp > 1 needs pp > 1
        (32, 2, None, True, False),  # overlap with pp > 1 needs vp >= 2
        (32, 2, 1, True, False),  # vp=1 is off, so it does not satisfy the overlap requirement
        (32, 2, 2, True, True),  # pipeline interleaved over 2 chunks
        (32, 4, 4, True, True),  # pipeline interleaved over 4 chunks
        (32, 2, None, False, True),  # without overlap, pp > 1 does not require a virtual pipeline
        (32, 2, 1, False, True),  # without overlap, vp=1 at pp > 1 is allowed
        (None, 2, 1, True, True),  # unset ep is not MoE
        (1, 2, 1, True, True),  # ep=1 is not MoE
    ],
)
def test_constraint_check_virtual_pipeline_and_moe_pipelining(
    ep: int | None,
    pp: int,
    vp: int | None,
    a2a: bool,
    res: bool,
    tmp_path: Path,
) -> None:
    tdef = _megatron_bridge_tdef()
    tdef.cmd_args.ep = ep
    tdef.cmd_args.pp = pp
    tdef.cmd_args.vp = vp
    tdef.cmd_args.moe_a2a_overlap = a2a
    tr = TestRun(name="tr", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)

    assert tdef.constraint_check(tr, None) is res
