# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

from pathlib import Path

import pytest

from cloudai.core import TestRun
from cloudai.systems.standalone import StandaloneSystem
from cloudai.workloads.aiconfig import (
    AiconfiguratorCmdArgs,
    AiconfiguratorStandaloneCommandGenStrategy,
    AiconfiguratorTestDefinition,
)
from cloudai.workloads.aiconfig.aiconfigurator import Agg, Disagg


def test_gen_exec_command_writes_repro_script_and_returns_bash(
    tmp_path: Path, standalone_system: StandaloneSystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.executable", "/tmp/python")

    tdef = AiconfiguratorTestDefinition(
        name="aiconfig",
        description="desc",
        test_template_name="Aiconfigurator",
        cmd_args=AiconfiguratorCmdArgs(
            model_name="LLAMA3.1_70B",
            system="h200_sxm",
            backend="trtllm",
            version="0.20.0",
            isl=4000,
            osl=500,
            disagg=Disagg(
                p_tp=1,
                p_pp=1,
                p_dp=1,
                p_bs=1,
                p_workers=1,
                d_tp=1,
                d_pp=1,
                d_dp=1,
                d_bs=8,
                d_workers=2,
                prefill_correction_scale=1.0,
                decode_correction_scale=1.0,
            ),
        ),
    )
    out_dir = tmp_path / "out"
    tr = TestRun(name="tr", test=tdef, num_nodes=1, nodes=[], output_path=out_dir)

    strategy = AiconfiguratorStandaloneCommandGenStrategy(standalone_system, tr)
    cmd = strategy.gen_exec_command()

    assert cmd.startswith("bash ")

    script_path = out_dir.resolve() / "run_simple_predictor.sh"
    assert script_path.is_file()

    content = script_path.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in content

    assert "/tmp/python" in content
    assert "-m cloudai.workloads.aiconfig.simple_predictor" in content
    assert "--model-name" in content and "LLAMA3.1_70B" in content
    assert "--system" in content and "h200_sxm" in content
    assert "--mode" in content and "disagg" in content
    assert "--d-bs" in content and "8" in content

    assert str((out_dir.resolve() / "report.json")) in content
    assert str((out_dir.resolve() / "stdout.txt")) in content
    assert str((out_dir.resolve() / "stderr.txt")) in content


def test_gen_exec_command_agg_branch(
    tmp_path: Path, standalone_system: StandaloneSystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.executable", "/tmp/python")

    tdef = AiconfiguratorTestDefinition(
        name="aiconfig",
        description="desc",
        test_template_name="Aiconfigurator",
        cmd_args=AiconfiguratorCmdArgs(
            model_name="LLAMA3.1_70B",
            system="h200_sxm",
            backend="trtllm",
            version="0.20.0",
            isl=4000,
            osl=500,
            agg=Agg(batch_size=8, ctx_tokens=16, tp=1, pp=1, dp=1, moe_tp=1, moe_ep=1),
        ),
    )
    out_dir = tmp_path / "out-agg"
    tr = TestRun(name="tr", test=tdef, num_nodes=1, nodes=[], output_path=out_dir)

    strategy = AiconfiguratorStandaloneCommandGenStrategy(standalone_system, tr)
    cmd = strategy.gen_exec_command()
    assert cmd.startswith("bash ")

    content = (out_dir.resolve() / "run_simple_predictor.sh").read_text(encoding="utf-8")
    assert "--mode" in content and "agg" in content
    assert "--batch-size" in content and "8" in content
    assert "--ctx-tokens" in content and "16" in content


def test_cmd_args_requires_exactly_one_mode() -> None:
    with pytest.raises(ValueError):
        AiconfiguratorCmdArgs(
            model_name="LLAMA3.1_70B",
            system="h200_sxm",
            isl=4000,
            osl=500,
        )


def test_cmd_args_rejects_both_agg_and_disagg() -> None:
    with pytest.raises(ValueError):
        AiconfiguratorCmdArgs(
            model_name="LLAMA3.1_70B",
            system="h200_sxm",
            isl=4000,
            osl=500,
            agg=Agg(batch_size=8, ctx_tokens=16, tp=1, pp=1, dp=1, moe_tp=1, moe_ep=1),
            disagg=Disagg(
                p_tp=1,
                p_pp=1,
                p_dp=1,
                p_bs=1,
                p_workers=1,
                d_tp=1,
                d_pp=1,
                d_dp=1,
                d_bs=8,
                d_workers=2,
            ),
        )
