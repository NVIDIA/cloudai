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
import yaml

from cloudai._core.test import Test
from cloudai._core.test_scenario import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.ai_dynamo import (
    AIDynamoArgs,
    AIDynamoCmdArgs,
    AIDynamoSlurmCommandGenStrategy,
    AIDynamoTestDefinition,
    CommonConfig,
    DecodeWorkerArgs,
    FrontendArgs,
    GenAIPerfArgs,
    PrefillWorkerArgs,
    SimpleLoadBalancerArgs,
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
            common=CommonConfig(
                **{
                    "model": "nvidia/Llama-3.1-405B-Instruct-FP8",
                    "kv-transfer-config": '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
                    "served_model_name": "nvidia/Llama-3.1-405B-Instruct-FP8",
                }
            ),
            frontend=FrontendArgs(
                endpoint="dynamo.SimpleLoadBalancer.generate_disagg",
                port=8000,
                port_etcd=1234,
                port_nats=5678,
            ),
            simple_load_balancer=SimpleLoadBalancerArgs(**{"enable_disagg": True}),
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
            endpoint="/chat",
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
    with pytest.raises(FileNotFoundError):
        _ = td.huggingface_home_host_path


def test_hugging_face_home_path_skip_validation(test_run: TestRun) -> None:
    td = cast(AIDynamoTestDefinition, test_run.test.test_definition)
    td.cmd_args.huggingface_home_host_path = Path("/nonexistent")
    td.cmd_args.skip_huggingface_home_host_path_validation = True
    # Should not raise an exception when validation is skipped
    path = td.huggingface_home_host_path
    assert path == Path("/nonexistent")


def test_container_mounts(strategy: AIDynamoSlurmCommandGenStrategy, test_run: TestRun) -> None:
    mounts = strategy._container_mounts()
    td = cast(AIDynamoTestDefinition, test_run.test.test_definition)
    script_host = test_run.output_path / "run.sh"
    yaml_config_path = test_run.output_path / "dynamo_config.yaml"
    assert mounts == [
        f"{td.huggingface_home_host_path}:{td.cmd_args.huggingface_home_container_path}",
        f"{script_host}:/opt/run.sh",
        f"{yaml_config_path}:{yaml_config_path}",
    ]
    assert script_host.exists()
    mode = script_host.stat().st_mode
    assert bool(mode & stat.S_IXUSR)


def test_yaml_config_generation(strategy: AIDynamoSlurmCommandGenStrategy, test_run: TestRun) -> None:
    td = cast(AIDynamoTestDefinition, test_run.test.test_definition)
    yaml_path = (test_run.output_path / "dynamo_config.yaml").resolve()
    yaml_path = strategy._generate_yaml_config(td, yaml_path)

    assert yaml_path.exists()

    with open(yaml_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
        expected_config = {
            "Common": {
                "model": "nvidia/Llama-3.1-405B-Instruct-FP8",
                "kv-transfer-config": '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
                "served_model_name": "nvidia/Llama-3.1-405B-Instruct-FP8",
            },
            "Frontend": {
                "common-configs": ["model", "kv-transfer-config", "served_model_name"],
                "endpoint": "dynamo.SimpleLoadBalancer.generate_disagg",
                "port": 8000,
            },
            "SimpleLoadBalancer": {
                "common-configs": ["model", "kv-transfer-config", "served_model_name"],
                "enable_disagg": True,
            },
            "VllmDecodeWorker": {
                "common-configs": ["model", "kv-transfer-config", "served_model_name"],
                "enforce-eager": True,
                "gpu-memory-utilization": 0.95,
                "tensor-parallel-size": 8,
                "pipeline-parallel-size": 1,
                "ServiceArgs": {
                    "workers": 1,
                    "resources": {
                        "gpu": "8",
                    },
                },
            },
            "VllmPrefillWorker": {
                "common-configs": ["model", "kv-transfer-config", "served_model_name"],
                "enforce-eager": True,
                "gpu-memory-utilization": 0.95,
                "tensor-parallel-size": 8,
                "pipeline-parallel-size": 1,
                "ServiceArgs": {
                    "workers": 1,
                    "resources": {
                        "gpu": "8",
                    },
                },
            },
        }
        assert config == expected_config


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
    result = strategy._dynamo_cmd(module, Path(config))
    assert result.strip() == expected
