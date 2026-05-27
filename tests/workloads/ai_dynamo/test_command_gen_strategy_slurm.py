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
from typing import cast

import pytest

from cloudai._core.test_scenario import TestRun
from cloudai.core import GitRepo
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.ai_dynamo import (
    AIDynamoArgs,
    AIDynamoCmdArgs,
    AIDynamoSlurmCommandGenStrategy,
    AIDynamoTestDefinition,
    AIPerf,
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


def test_gen_script_args_contains_aiperf_accuracy_args(strategy: AIDynamoSlurmCommandGenStrategy) -> None:
    td = cast(AIDynamoTestDefinition, strategy.test_run.test)
    td.cmd_args.workloads = "aiperf.sh"
    setup_cmd = "python -m pip install --break-system-packages --upgrade 'aiperf[accuracy]==0.8.0'"
    extra_inputs = '{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
    td.cmd_args.aiperf = AIPerf.model_validate(
        {
            "setup-cmd": setup_cmd,
            "args": {
                "accuracy-benchmark": "mmlu",
                "accuracy-n-shots": 5,
                "accuracy-tasks": "abstract_algebra",
                "concurrency": 10,
                "extra-inputs": extra_inputs,
                "num-requests": 100,
            },
        }
    )

    result = strategy._gen_script_args(td)

    assert f'--aiperf-setup-cmd "{setup_cmd}"' in result
    assert '--aiperf-args-accuracy-benchmark "mmlu"' in result
    assert '--aiperf-args-accuracy-n-shots "5"' in result
    assert '--aiperf-args-accuracy-tasks "abstract_algebra"' in result
    assert '--aiperf-args-concurrency "10"' in result
    assert f"--aiperf-args-extra-inputs '{extra_inputs}'" in result
    assert '--aiperf-args-num-requests "100"' in result


def test_gen_script_args_quotes_worker_json_args(strategy: AIDynamoSlurmCommandGenStrategy) -> None:
    td = cast(AIDynamoTestDefinition, strategy.test_run.test)
    config = '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
    td.cmd_args.dynamo.prefill_worker.args = WorkerBaseArgs.model_validate({"kv-transfer-config": config})
    td.cmd_args.dynamo.decode_worker.args = WorkerBaseArgs.model_validate({"kv-transfer-config": config})

    result = strategy._gen_script_args(td)

    assert f"--prefill-args-kv-transfer-config '{config}'" in result
    assert f"--decode-args-kv-transfer-config '{config}'" in result
