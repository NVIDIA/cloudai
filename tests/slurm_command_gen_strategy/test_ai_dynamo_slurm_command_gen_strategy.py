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
from typing import cast
from unittest.mock import Mock

import pytest

from cloudai._core.test import Test
from cloudai._core.test_scenario import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.ai_dynamo import (
    AIDynamoArgs,
    AIDynamoCmdArgs,
    AIDynamoSlurmCommandGenStrategy,
    AIDynamoTestDefinition,
    DecodeWorkerArgs,
    GenAIPerfArgs,
    PrefillWorkerArgs,
)


@pytest.fixture
def cmd_args() -> AIDynamoCmdArgs:
    return AIDynamoCmdArgs(
        docker_image_url="url",
        huggingface_home_host_path=Path.home() / ".cache/huggingface",
        huggingface_home_container_path=Path("/root/.cache/huggingface"),
        extra_args="",
        node_setup_cmd="",
        dynamo=AIDynamoArgs(
            model="nvidia/Llama-3.1-405B-Instruct-FP8",
            port=8000,
            etcd_port=1234,
            nats_port=5678,
            prefill_worker=PrefillWorkerArgs(
                **{
                    "num_nodes": 1,
                    "gpu-memory-utilization": 0.95,
                    "tensor-parallel-size": 8,
                    "ServiceArgs": {"workers": 1, "resources": {"gpu": "8"}},
                }
            ),
            decode_worker=DecodeWorkerArgs(
                **{
                    "num_nodes": 1,
                    "gpu-memory-utilization": 0.95,
                    "tensor-parallel-size": 8,
                    "ServiceArgs": {"workers": 1, "resources": {"gpu": "8"}},
                }
            ),
        ),
        sleep_seconds=100,
        genai_perf=GenAIPerfArgs(
            endpoint_type="chat",
            streaming=True,
            warmup_request_count=10,
            random_seed=42,
            synthetic_input_tokens_mean=128,
            synthetic_input_tokens_stddev=32,
            output_tokens_mean=256,
            output_tokens_stddev=64,
            extra_inputs=None,
            concurrency=2,
            request_count=10,
        ),
    )


@pytest.fixture
def test_run(tmp_path: Path, cmd_args: AIDynamoCmdArgs) -> TestRun:
    hf_home = tmp_path / "huggingface"
    hf_home.mkdir()
    cmd_args.huggingface_home_host_path = hf_home
    tdef = AIDynamoTestDefinition(
        name="test",
        description="desc",
        test_template_name="template",
        cmd_args=cmd_args,
    )
    test = Test(test_definition=tdef, test_template=Mock())
    return TestRun(name="run", test=test, nodes=["n0", "n1"], num_nodes=2, output_path=tmp_path)


@pytest.fixture
def strategy(slurm_system: SlurmSystem, test_run: TestRun) -> AIDynamoSlurmCommandGenStrategy:
    return AIDynamoSlurmCommandGenStrategy(slurm_system, test_run)


def test_hugging_face_home_path_valid(test_run: TestRun) -> None:
    td = cast(AIDynamoTestDefinition, test_run.test.test_definition)
    path = td.huggingface_home_host_path
    assert path.exists()
    assert path.is_dir()


def test_hugging_face_home_path_missing(test_run: TestRun) -> None:
    td = cast(AIDynamoTestDefinition, test_run.test.test_definition)
    td.cmd_args.huggingface_home_host_path = Path("/nonexistent")
    with pytest.raises(FileNotFoundError, match="HuggingFace home path not found at /nonexistent"):
        _ = td.huggingface_home_host_path


def test_container_mounts(strategy: AIDynamoSlurmCommandGenStrategy, test_run: TestRun) -> None:
    mounts = strategy._container_mounts()
    td = cast(AIDynamoTestDefinition, test_run.test.test_definition)
    script_host = test_run.output_path / "run.sh"
    assert mounts == [
        f"{td.huggingface_home_host_path}:{td.cmd_args.huggingface_home_container_path}",
        f"{script_host}:/opt/run.sh",
    ]
    assert script_host.exists()
    mode = script_host.stat().st_mode
    assert bool(mode & stat.S_IXUSR)


def test_default_worker_cmd(
    strategy: AIDynamoSlurmCommandGenStrategy,
    test_run: TestRun,
) -> None:
    td = cast(AIDynamoTestDefinition, test_run.test.test_definition)
    worker = DecodeWorkerArgs(
        num_nodes=1,
        **{
            "ServiceArgs": {"workers": 1, "resources": {"gpu": "8"}},
            "gpu-memory-utilization": 0.95,
            "tensor-parallel-size": 8,
            "pipeline-parallel-size": 1,
            "data-parallel-size": 1,
            "enable-expert-parallel": False,
            "enforce-eager": True,
        },
    )
    result = strategy._dynamo_cmd(td, worker)
    assert "python3 components/main.py" in result


@pytest.mark.parametrize(
    "expected",
    [
        "dynamo serve graphs.agg:Frontend -f cfg.yaml",
        "dynamo serve components.worker:VllmPrefillWorker -f prefill.yaml",
        "dynamo serve components.worker:VllmDecodeWorker -f decode.yaml",
    ],
)
def test_dynamo_cmd(
    strategy: AIDynamoSlurmCommandGenStrategy,
    test_run: TestRun,
    expected: str,
) -> None:
    td = cast(AIDynamoTestDefinition, test_run.test.test_definition)
    worker = DecodeWorkerArgs(
        num_nodes=1,
        cmd=expected,
        **{
            "ServiceArgs": {"workers": 1, "resources": {"gpu": "8"}},
            "gpu-memory-utilization": 0.95,
            "tensor-parallel-size": 8,
            "pipeline-parallel-size": 1,
            "data-parallel-size": 1,
            "enable-expert-parallel": False,
            "enforce-eager": True,
        },
    )
    result = strategy._dynamo_cmd(td, worker)
    assert expected in result
