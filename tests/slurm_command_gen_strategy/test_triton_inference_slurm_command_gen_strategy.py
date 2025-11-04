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

import stat
from pathlib import Path
from typing import List, cast
from unittest.mock import Mock

import pytest

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.triton_inference import (
    TritonInferenceCmdArgs,
    TritonInferenceSlurmCommandGenStrategy,
    TritonInferenceTestDefinition,
)


@pytest.fixture
def test_run(tmp_path: Path) -> TestRun:
    args = TritonInferenceCmdArgs(
        server_docker_image_url="nvcr.io/nim/deepseek-ai/deepseek-r1:1.7.2",
        client_docker_image_url="nvcr.io/nvidia/tritonserver:25.01-py3-sdk",
        served_model_name="model",
        tokenizer="tok",
    )
    nim_path = tmp_path / "nim"
    nim_path.mkdir(parents=True, exist_ok=True)
    tdef = TritonInferenceTestDefinition(
        name="dsr1",
        description="desc",
        test_template_name="tt",
        cmd_args=args,
        extra_env_vars={
            "NIM_MODEL_NAME": str(nim_path),
            "NIM_CACHE_PATH": str(nim_path),
        },
    )

    return TestRun(name="run", test=tdef, nodes=["nodeA", "nodeB", "nodeC"], num_nodes=3)


@pytest.fixture
def strategy(slurm_system: SlurmSystem, test_run: TestRun) -> TritonInferenceSlurmCommandGenStrategy:
    return TritonInferenceSlurmCommandGenStrategy(slurm_system, test_run)


def test_container_mounts_invalid_model(tmp_path: Path, strategy: TritonInferenceSlurmCommandGenStrategy) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    strategy.test_run.test.extra_env_vars = {
        "NIM_MODEL_NAME": str(tmp_path / "nope"),
        "NIM_CACHE_PATH": str(cache_dir),
    }
    with pytest.raises(FileNotFoundError):
        strategy._container_mounts()


def test_container_mounts_with_model_and_cache(
    tmp_path: Path, strategy: TritonInferenceSlurmCommandGenStrategy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        TestRun,
        "output_path",
        property(lambda self: tmp_path),
        raising=True,
    )

    model_dir = tmp_path / "model"
    cache_dir = tmp_path / "cache"
    model_dir.mkdir()
    cache_dir.mkdir()

    tdef = cast(TritonInferenceTestDefinition, strategy.test_run.test)
    tdef.extra_env_vars["NIM_MODEL_NAME"] = str(model_dir)
    tdef.extra_env_vars["NIM_CACHE_PATH"] = str(cache_dir)

    mounts = strategy._container_mounts()

    expected = [
        f"{model_dir}:{model_dir}:ro",
        f"{cache_dir}:{cache_dir}:rw",
        f"{tmp_path / 'start_server_wrapper.sh'}:/opt/nim/start_server_wrapper.sh:ro",
    ]

    assert mounts == expected


def test_generate_start_wrapper_script(tmp_path: Path, strategy: TritonInferenceSlurmCommandGenStrategy) -> None:
    script_path = tmp_path / "script.sh"
    env_vars = {"KEY": "VALUE", "NIM_LEADER_IP_ADDRESS": "X", "NIM_NODE_RANK": "Y"}
    strategy._generate_start_wrapper_script(script_path, env_vars)
    content = script_path.read_text().splitlines()
    assert content[0] == "#!/bin/bash"
    assert "export NIM_LEADER_IP_ADDRESS=${SLURM_JOB_MASTER_NODE}" in content
    assert "export KEY='VALUE'" in content
    mode = script_path.stat().st_mode
    assert bool(mode & stat.S_IXUSR)


def test_append_sbatch_directives(strategy: TritonInferenceSlurmCommandGenStrategy) -> None:
    lines: List[str] = []
    strategy._append_sbatch_directives(lines)
    assert "export HEAD_NODE=$SLURM_JOB_MASTER_NODE" in lines
    assert "export NIM_LEADER_IP_ADDRESS=$SLURM_JOB_MASTER_NODE" in lines
    assert "export NIM_NUM_COMPUTE_NODES=2" in lines
    assert "export NIM_MODEL_TOKENIZER='deepseek-ai/DeepSeek-R1'" in lines


def test_build_server_srun(strategy: TritonInferenceSlurmCommandGenStrategy) -> None:
    strategy.gen_srun_prefix = Mock(return_value=["P"])
    strategy.gen_nsys_command = Mock(return_value=["N"])

    result = strategy._build_server_srun(2)

    parts = result.split()
    assert parts[:2] == ["P", "--nodes=2"]
    assert "--ntasks=2" in parts
    assert "--ntasks-per-node=1" in parts
    assert parts[-2:] == ["N", "/opt/nim/start_server_wrapper.sh"]


@pytest.mark.parametrize("streaming", [True, False])
def test_build_client_srun(strategy: TritonInferenceSlurmCommandGenStrategy, streaming: bool) -> None:
    strategy.test_run.test.cmd_args.streaming = streaming
    result = strategy._build_client_srun(1)
    assert strategy.test_run.test.cmd_args.client_docker_image_url in result
    assert "--nodes=1" in result
    assert "--ntasks=1" in result
    assert "genai-perf" in result
    if streaming:
        assert "--streaming" in result
    else:
        assert "--streaming" not in result


def test_gen_srun_command(strategy: TritonInferenceSlurmCommandGenStrategy) -> None:
    strategy._build_server_srun = Mock(return_value="S")
    strategy._build_client_srun = Mock(return_value="C")
    cmd = strategy._gen_srun_command()
    assert cmd == f"S &\n\nsleep {strategy.test_run.test.cmd_args.sleep_seconds}\n\nC"
