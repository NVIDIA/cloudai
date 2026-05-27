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

import subprocess
from pathlib import Path
from typing import cast

import pytest

from cloudai._core.test_scenario import TestRun
from cloudai.core import GitRepo
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.ai_dynamo import (
    AIDynamoArgs,
    AIDynamoCmdArgs,
    AIDynamoSemanticEvalCmdArgs,
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


def test_gen_script_args_contains_semantic_eval_args(strategy: AIDynamoSlurmCommandGenStrategy, test_run: TestRun):
    tdef = cast(AIDynamoTestDefinition, test_run.test)
    tdef.semantic_eval_cmd_args = AIDynamoSemanticEvalCmdArgs(
        module="sglang.test.run_eval",
        args="--host {host} --port {port} --model {model}",
        log_file="semantic_eval.log",
    )
    tdef.semantic_eval_cmd_args.script.installed_path = Path("/tmp/install/semantic_eval.sh")

    args = strategy._gen_script_args(tdef)

    assert '--semantic_eval-script "/cloudai_install/semantic_eval.sh"' in args
    assert '--semantic_eval-module "sglang.test.run_eval"' in args
    assert '--semantic_eval-args "--host {host} --port {port} --model {model}"' in args
    assert '--semantic_eval-log-file "semantic_eval.log"' in args


def test_semantic_eval_script_writes_log_without_metric_report(tmp_path: Path) -> None:
    module = tmp_path / "semantic_eval.py"
    module.write_text("print('Accuracy: 0.42')\n", encoding="utf-8")
    script = Path("src/cloudai/workloads/ai_dynamo/semantic_eval.sh")

    result = subprocess.run(
        [
            str(script),
            "--result-dir",
            str(tmp_path),
            "--module",
            str(module),
            "--args",
            "--host {host} --port {port} --model {model}",
            "--model",
            "Qwen/Qwen3-0.6B",
            "--url",
            "http://localhost",
            "--port",
            "8000",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert (tmp_path / "semantic_eval.log").read_text(encoding="utf-8").strip() == "Accuracy: 0.42"
    assert not (tmp_path / "semantic_eval_report.csv").exists()


def test_semantic_eval_script_returns_command_failure(tmp_path: Path) -> None:
    module = tmp_path / "semantic_eval.py"
    module.write_text("raise SystemExit(7)\n", encoding="utf-8")
    script = Path("src/cloudai/workloads/ai_dynamo/semantic_eval.sh")

    result = subprocess.run(
        [str(script), "--result-dir", str(tmp_path), "--module", str(module)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 7
