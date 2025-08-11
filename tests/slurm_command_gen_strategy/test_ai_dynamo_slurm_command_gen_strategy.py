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
        dynamo=AIDynamoArgs(
            prefill_worker=PrefillWorkerArgs(
                **{
                    "num-nodes": 1,
                    "gpu-memory-utilization": 0.95,
                    "tensor-parallel-size": 8,
                    "ServiceArgs": {"workers": 1, "resources": {"gpu": "8"}},
                }
            ),
            decode_worker=DecodeWorkerArgs(
                **{
                    "num-nodes": 1,
                    "gpu-memory-utilization": 0.95,
                    "tensor-parallel-size": 8,
                    "ServiceArgs": {"workers": 1, "resources": {"gpu": "8"}},
                }
            ),
        ),
        genai_perf=GenAIPerfArgs(
            **{
                "endpoint-type": "chat",
                "streaming": True,
                "warmup-request-count": 10,
                "random-seed": 42,
                "synthetic-input-tokens-mean": 128,
                "synthetic-input-tokens-stddev": 32,
                "output-tokens-mean": 256,
                "output-tokens-stddev": 64,
                "extra-inputs": None,
                "concurrency": 2,
                "request-count": 10,
            }
        ),
    )


@pytest.fixture
def test_run(tmp_path: Path, cmd_args: AIDynamoCmdArgs) -> TestRun:
    hf_home = tmp_path / "huggingface"
    hf_home.mkdir()
    cmd_args.huggingface_home_host_path = hf_home

    dynamo_repo_path = tmp_path / "dynamo_repo"
    dynamo_repo_path.mkdir()

    tdef = AIDynamoTestDefinition(
        name="test",
        description="desc",
        test_template_name="template",
        cmd_args=cmd_args,
    )
    tdef.dynamo_repo.installed_path = dynamo_repo_path

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
    with pytest.raises(FileNotFoundError):
        _ = td.huggingface_home_host_path


def test_container_mounts(strategy: AIDynamoSlurmCommandGenStrategy, test_run: TestRun) -> None:
    mounts = strategy._container_mounts()
    td = cast(AIDynamoTestDefinition, test_run.test.test_definition)
    dynamo_repo_path = td.dynamo_repo.installed_path
    assert dynamo_repo_path is not None, "dynamo_repo_path should be set in the test fixture"

    assert mounts == [
        f"{dynamo_repo_path!s}:{dynamo_repo_path!s}",
        f"{td.cmd_args.huggingface_home_host_path!s}:{td.cmd_args.huggingface_home_container_path!s}",
        f"{td.script.installed_path.absolute()!s}:{td.script.installed_path.absolute()!s}",
    ]


@pytest.mark.parametrize(
    "module, config, expected",
    [
        ("graphs.agg:Frontend", "cfg.yaml", "dynamo serve graphs.agg:Frontend -f cfg.yaml"),
        (
            "components.worker:VllmPrefillWorker",
            "prefill.yaml",
            "dynamo serve components.worker:VllmPrefillWorker -f prefill.yaml",
        ),
        (
            "components.worker:VllmDecodeWorker",
            "decode.yaml",
            "dynamo serve components.worker:VllmDecodeWorker -f decode.yaml",
        ),
    ],
)
def test_dynamo_cmd(
    strategy: AIDynamoSlurmCommandGenStrategy,
    module: str,
    config: str,
    expected: str,
) -> None:
    result = strategy.gen_dynamo_cmd(module, Path(config))
    assert result.strip() == expected
