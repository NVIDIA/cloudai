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
from cloudai.systems import SlurmSystem
from cloudai.workloads.ai_dynamo import (
    AIDynamoCmdArgs,
    AIDynamoSlurmCommandGenStrategy,
    AIDynamoTestDefinition,
)


@pytest.fixture
def strategy(slurm_system: SlurmSystem) -> AIDynamoSlurmCommandGenStrategy:
    return AIDynamoSlurmCommandGenStrategy(slurm_system, {})


@pytest.fixture
def cmd_args() -> AIDynamoCmdArgs:
    return AIDynamoCmdArgs(
        docker_image_url="url",
        served_model_name="gpt",
        num_prefill_nodes=1,
        num_decode_nodes=1,
        port_etcd=1234,
        port_nats=5678,
        config_path="config.yaml",
        port=8080,
        endpoint_type="chat",
        service_kind="openai",
        endpoint="/chat",
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
        sleep_seconds=100,
    )


@pytest.fixture
def test_run(tmp_path: Path, cmd_args: AIDynamoCmdArgs) -> TestRun:
    home = tmp_path / "huggingface"
    home.mkdir()
    tdef = AIDynamoTestDefinition(
        name="test",
        description="desc",
        test_template_name="template",
        cmd_args=cmd_args,
        extra_env_vars={"HF_HOME": str(home)},
    )
    test = Test(test_definition=tdef, test_template=Mock())
    return TestRun(name="run", test=test, nodes=["n0", "n1", "n2"], num_nodes=3, output_path=tmp_path)


def test_hugging_face_home_path_valid(test_run: TestRun) -> None:
    td = cast(AIDynamoTestDefinition, test_run.test.test_definition)
    path = td.hugging_face_home_path
    assert path.exists()
    assert path.is_dir()


def test_hugging_face_home_path_missing(test_run: TestRun) -> None:
    td = cast(AIDynamoTestDefinition, test_run.test.test_definition)
    td.extra_env_vars["HF_HOME"] = ""
    with pytest.raises(ValueError):
        _ = td.hugging_face_home_path


def test_container_mounts(strategy: AIDynamoSlurmCommandGenStrategy, test_run: TestRun) -> None:
    mounts = strategy._container_mounts(test_run)
    td = cast(AIDynamoTestDefinition, test_run.test.test_definition)
    script_host = test_run.output_path / "run.sh"
    assert mounts == [
        f"{td.hugging_face_home_path}:{td.hugging_face_home_path}",
        f"{script_host}:/opt/run.sh",
    ]
    assert script_host.exists()
    mode = script_host.stat().st_mode
    assert bool(mode & stat.S_IXUSR)


@pytest.mark.parametrize(
    "module, config, service_name, expected",
    [
        ("graphs.agg_router:Frontend", "cfg.yaml", None, "dynamo serve graphs.agg_router:Frontend -f cfg.yaml"),
        (
            "components.prefill_worker:PrefillWorker",
            "prefill.yaml",
            None,
            "dynamo serve components.prefill_worker:PrefillWorker -f prefill.yaml",
        ),
        (
            "components.worker:VllmWorker",
            "decode.yaml",
            "VllmWorker",
            "dynamo serve components.worker:VllmWorker -f decode.yaml --service-name VllmWorker",
        ),
    ],
)
def test_dynamo_cmd(
    strategy: AIDynamoSlurmCommandGenStrategy,
    module: str,
    config: str,
    service_name: str | None,
    expected: str,
) -> None:
    result = strategy._dynamo_cmd(module, config, service_name)
    assert result.strip() == expected
