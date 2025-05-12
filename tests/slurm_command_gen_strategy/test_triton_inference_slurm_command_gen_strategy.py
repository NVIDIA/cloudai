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
from typing import Any, Dict, List, cast
from unittest.mock import Mock

import pytest

from cloudai import Test, TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.triton_inference import (
    TritonInferenceCmdArgs,
    TritonInferenceSlurmCommandGenStrategy,
    TritonInferenceTestDefinition,
)


@pytest.fixture
def strategy(slurm_system: SlurmSystem) -> TritonInferenceSlurmCommandGenStrategy:
    return TritonInferenceSlurmCommandGenStrategy(slurm_system, {})


@pytest.fixture
def test_run(tmp_path: Path) -> TestRun:
    args = TritonInferenceCmdArgs(
        server_docker_image_url="nvcr.io/nim/deepseek-ai/deepseek-r1:1.7.2",
        client_docker_image_url="nvcr.io/nvidia/tritonserver:25.01-py3-sdk",
        served_model_name="model",
        tokenizer="tok",
    )
    tdef = TritonInferenceTestDefinition(
        name="dsr1",
        description="desc",
        test_template_name="tt",
        cmd_args=args,
        extra_env_vars={},
    )
    test = Test(test_definition=tdef, test_template=Mock())
    return TestRun(name="run", test=test, nodes=["nodeA", "nodeB", "nodeC"], num_nodes=3)


def test_container_mounts_invalid_model(
    tmp_path: Path,
    strategy: TritonInferenceSlurmCommandGenStrategy,
) -> None:
    args = TritonInferenceCmdArgs(
        server_docker_image_url="nvcr.io/nim/deepseek-ai/deepseek-r1:1.7.2",
        client_docker_image_url="nvcr.io/nvidia/tritonserver:25.01-py3-sdk",
        served_model_name="m",
        tokenizer="tok",
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    tdef = TritonInferenceTestDefinition(
        name="x",
        description="",
        test_template_name="",
        cmd_args=args,
        extra_env_vars={
            "NIM_MODEL_NAME": str(tmp_path / "nope"),
            "NIM_CACHE_PATH": str(cache_dir),
        },
    )
    test = Test(test_definition=tdef, test_template=Mock())
    tr = TestRun(name="run", test=test, nodes=[], num_nodes=1)
    with pytest.raises(FileNotFoundError):
        strategy._container_mounts(tr)


def test_container_mounts_with_model_and_cache(
    tmp_path: Path,
    strategy: TritonInferenceSlurmCommandGenStrategy,
    test_run: TestRun,
    monkeypatch: pytest.MonkeyPatch,
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

    tdef = cast(TritonInferenceTestDefinition, test_run.test.test_definition)
    tdef.extra_env_vars["NIM_MODEL_NAME"] = str(model_dir)
    tdef.extra_env_vars["NIM_CACHE_PATH"] = str(cache_dir)

    mounts = strategy._container_mounts(test_run)

    expected = [
        f"{model_dir}:{model_dir}:ro",
        f"{cache_dir}:{cache_dir}:rw",
        f"{tmp_path / 'start_server_wrapper.sh'}:/opt/nim/start_server_wrapper.sh:ro",
    ]

    assert mounts == expected


def test_generate_start_wrapper_script(
    tmp_path: Path,
    strategy: TritonInferenceSlurmCommandGenStrategy,
) -> None:
    script_path = tmp_path / "script.sh"
    env_vars = {"KEY": "VALUE", "NIM_LEADER_IP_ADDRESS": "X", "NIM_NODE_RANK": "Y"}
    strategy._generate_start_wrapper_script(script_path, env_vars)
    content = script_path.read_text().splitlines()
    assert content[0] == "#!/bin/bash"
    assert "export NIM_LEADER_IP_ADDRESS=${SLURM_JOB_MASTER_NODE}" in content
    assert "export KEY='VALUE'" in content
    mode = script_path.stat().st_mode
    assert bool(mode & stat.S_IXUSR)


def test_append_sbatch_directives(strategy: TritonInferenceSlurmCommandGenStrategy, test_run: TestRun) -> None:
    args: Dict[str, Any] = {"num_nodes": 3, "node_list_str": ""}
    lines: List[str] = []
    strategy._append_sbatch_directives(lines, args, test_run)
    assert "export HEAD_NODE=$SLURM_JOB_MASTER_NODE" in lines
    assert "export NIM_LEADER_IP_ADDRESS=$SLURM_JOB_MASTER_NODE" in lines
    assert "export NIM_NUM_COMPUTE_NODES=2" in lines
    assert "export NIM_MODEL_TOKENIZER='deepseek-ai/DeepSeek-R1'" in lines


def test_build_server_srun(
    strategy: TritonInferenceSlurmCommandGenStrategy,
) -> None:
    strategy.gen_srun_prefix = Mock(return_value=["P"])
    strategy.gen_nsys_command = Mock(return_value=["N"])
    tdef = TritonInferenceTestDefinition(
        name="z",
        description="",
        test_template_name="",
        cmd_args=TritonInferenceCmdArgs(
            server_docker_image_url="nvcr.io/nim/deepseek-ai/deepseek-r1:1.7.2",
            client_docker_image_url="nvcr.io/nvidia/tritonserver:25.01-py3-sdk",
            served_model_name="",
            tokenizer="",
        ),
        extra_env_vars={"NIM_CACHE_PATH": "/tmp"},
    )
    test = Test(test_definition=tdef, test_template=Mock())
    tr = TestRun(name="run", test=test, nodes=["n1", "n2", "n3"], num_nodes=3)
    result = strategy._build_server_srun({}, tr, 2)
    parts = result.split()
    assert parts[:2] == ["P", "--nodes=2"]
    assert "--ntasks=2" in parts
    assert "--ntasks-per-node=1" in parts
    assert parts[-2:] == ["N", "/opt/nim/start_server_wrapper.sh"]


@pytest.mark.parametrize("streaming", [True, False])
def test_build_client_srun(
    strategy: TritonInferenceSlurmCommandGenStrategy,
    test_run: TestRun,
    streaming: bool,
) -> None:
    strategy.gen_srun_prefix = Mock(return_value=["C"])
    test_run.test.test_definition.cmd_args.streaming = streaming
    result = strategy._build_client_srun({}, test_run, 1)
    called_args, called_tr = strategy.gen_srun_prefix.call_args.args
    assert called_args["image_path"] == test_run.test.test_definition.cmd_args.client_docker_image_url
    assert called_tr is test_run
    parts = result.split()
    assert "--nodes=1" in parts
    assert "--ntasks=1" in parts
    assert "genai-perf" in parts
    if streaming:
        assert "--streaming" in parts
    else:
        assert "--streaming" not in parts


def test_gen_srun_command(
    strategy: TritonInferenceSlurmCommandGenStrategy,
    test_run: TestRun,
) -> None:
    strategy._build_server_srun = Mock(return_value="S")
    strategy._build_client_srun = Mock(return_value="C")
    cmd = strategy._gen_srun_command({}, {}, {}, test_run)
    assert cmd == f"S &\n\nsleep {test_run.test.test_definition.cmd_args.sleep_seconds}\n\nC"
