# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai._core.test_scenario import TestRun
from cloudai.core import GitRepo
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.ai_dynamo import (
    AIDynamoArgs,
    AIDynamoCmdArgs,
    AIDynamoSlurmCommandGenStrategy,
    AIDynamoTestDefinition,
    GenAIPerf,
    LMCache,
    LMCacheArgs,
    WorkerBaseArgs,
    WorkerConfig,
)


@pytest.fixture
def cmd_args() -> AIDynamoCmdArgs:
    return AIDynamoCmdArgs(
        docker_image_url="url",
        dynamo=AIDynamoArgs(
            model="model",
            workspace_path="/workspace",
            prefill_worker=WorkerConfig(
                cmd="python3 -m dynamo.vllm --is-prefill-worker",
                worker_initialized_regex="VllmWorker.*has.been.initialized",
                **{
                    "num-nodes": 1,
                    "args": WorkerBaseArgs(
                        **{
                            "gpu-memory-utilization": 0.95,
                            "tensor-parallel-size": 8,
                        }
                    ),
                },
            ),
            decode_worker=WorkerConfig(
                cmd="python3 -m dynamo.vllm",
                worker_initialized_regex="VllmWorker.*has.been.initialized",
                **{
                    "num-nodes": 1,
                    "args": WorkerBaseArgs(
                        **{
                            "gpu-memory-utilization": 0.95,
                            "tensor-parallel-size": 8,
                        }
                    ),
                },
            ),
        ),
        genai_perf=GenAIPerf(
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
        lmcache=LMCache(args=LMCacheArgs()),
    )


@pytest.fixture
def test_run(tmp_path: Path, cmd_args: AIDynamoCmdArgs) -> TestRun:
    dynamo_repo_path = tmp_path / "dynamo_repo"
    dynamo_repo_path.mkdir()

    tdef = AIDynamoTestDefinition(
        name="test",
        description="desc",
        test_template_name="template",
        cmd_args=cmd_args,
        repo=GitRepo(
            url="https://github.com/ai-dynamo/dynamo.git",
            commit="f7e468c7e8ff0d1426db987564e60572167e8464",
            installed_path=dynamo_repo_path,
        ),
    )

    return TestRun(name="run", test=tdef, nodes=["n0", "n1"], num_nodes=2, output_path=tmp_path)


@pytest.fixture
def strategy(slurm_system: SlurmSystem, test_run: TestRun) -> AIDynamoSlurmCommandGenStrategy:
    return AIDynamoSlurmCommandGenStrategy(slurm_system, test_run)


def test_container_mounts(strategy: AIDynamoSlurmCommandGenStrategy, test_run: TestRun) -> None:
    mounts = strategy._container_mounts()

    td = test_run.test
    expected = [
        f"{strategy.system.hf_home_path.absolute()}:{strategy.CONTAINER_MOUNT_HF_HOME}",
    ]
    if td.cmd_args.storage_cache_dir:
        expected.append(f"{td.cmd_args.storage_cache_dir}:{td.cmd_args.storage_cache_dir}")
    assert mounts == expected


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
