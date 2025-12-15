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

from typing import Any, cast

import pytest
import yaml
from pydantic import BaseModel

from cloudai.core import TestRun
from cloudai.systems.kubernetes import KubernetesSystem
from cloudai.workloads.ai_dynamo import (
    AIDynamoArgs,
    AIDynamoCmdArgs,
    AIDynamoKubernetesJsonGenStrategy,
    AIDynamoTestDefinition,
    DecodeWorkerArgs,
    GenAIPerfArgs,
    PrefillWorkerArgs,
)


@pytest.fixture(params=["agg", "disagg"])
def dynamo(request: Any) -> AIDynamoTestDefinition:
    dynamo = AIDynamoTestDefinition(
        name="test_dynamo",
        description="Test AI Dynamo workload",
        test_template_name="AIDynamo",
        cmd_args=AIDynamoCmdArgs(
            docker_image_url="nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1.post1",
            dynamo=AIDynamoArgs(
                decode_worker=DecodeWorkerArgs(
                    num_nodes=2, data_parallel_size=1, tensor_parallel_size=1, extra_args="--extra-decode-arg v"
                )
            ),
            genai_perf=GenAIPerfArgs(),
        ),
    )
    if request.param == "disagg":
        dynamo.cmd_args.dynamo.prefill_worker = PrefillWorkerArgs(
            num_nodes=3, tensor_parallel_size=1, extra_args="--extra-prefill-arg v"
        )

    return dynamo


@pytest.fixture
def json_gen(
    k8s_system: KubernetesSystem, base_tr: TestRun, dynamo: AIDynamoTestDefinition
) -> AIDynamoKubernetesJsonGenStrategy:
    base_tr.test = dynamo
    return AIDynamoKubernetesJsonGenStrategy(k8s_system, base_tr)


def test_gen_frontend(json_gen: AIDynamoKubernetesJsonGenStrategy) -> None:
    system = cast(KubernetesSystem, json_gen.system)
    tdef = cast(AIDynamoTestDefinition, json_gen.test_run.test)

    frontend = json_gen.gen_frontend_dict()
    assert frontend.get("dynamoNamespace") == system.default_namespace
    assert frontend.get("componentType") == "frontend"
    assert frontend.get("replicas") == 1
    assert frontend.get("extraPodSpec", {}).get("mainContainer", {}).get("image") == tdef.cmd_args.docker_image_url


def dynamo_args_dict(model: BaseModel) -> dict:
    return model.model_dump(exclude_none=True, exclude={"num_nodes", "extra_args"})


def test_gen_decode(json_gen: AIDynamoKubernetesJsonGenStrategy) -> None:
    system = cast(KubernetesSystem, json_gen.system)
    tdef = cast(AIDynamoTestDefinition, json_gen.test_run.test)

    decode = json_gen.gen_decode_dict()
    assert decode.get("dynamoNamespace") == system.default_namespace
    assert decode.get("componentType") == "worker"
    assert decode.get("replicas") == 1

    args = ["--model", tdef.cmd_args.dynamo.model]
    if tdef.cmd_args.dynamo.prefill_worker:
        assert decode.get("subComponentType") == "decode-worker"
        args.append("--is-decode-worker")

    for arg, value in dynamo_args_dict(tdef.cmd_args.dynamo.decode_worker).items():
        args.extend([json_gen._to_dynamo_arg(arg), str(value)])
    if tdef.cmd_args.dynamo.decode_worker.extra_args:
        args.append(f"{tdef.cmd_args.dynamo.decode_worker.extra_args}")

    main_container = decode.get("extraPodSpec", {}).get("mainContainer", {})
    assert main_container.get("image") == tdef.cmd_args.docker_image_url
    assert main_container.get("workingDir") == tdef.cmd_args.dynamo.workspace_path
    assert main_container.get("command") == tdef.cmd_args.dynamo.decode_cmd.split()
    assert main_container.get("args") == args

    resources = decode.get("resources", {})
    assert resources.get("limits", {}).get("gpu") == f"{system.gpus_per_node}"


@pytest.mark.parametrize("num_nodes", [1, 2, 4])
def test_gen_decode_num_nodes(num_nodes: int, json_gen: AIDynamoKubernetesJsonGenStrategy) -> None:
    tdef = cast(AIDynamoTestDefinition, json_gen.test_run.test)
    tdef.cmd_args.dynamo.decode_worker.num_nodes = num_nodes

    decode = json_gen.gen_decode_dict()

    if num_nodes > 1:
        multinode = decode.get("multinode", {})
        assert multinode.get("nodeCount") == num_nodes
    else:
        assert "multinode" not in decode


def test_gen_prefill(json_gen: AIDynamoKubernetesJsonGenStrategy) -> None:
    system = cast(KubernetesSystem, json_gen.system)
    tdef = cast(AIDynamoTestDefinition, json_gen.test_run.test)

    if not tdef.cmd_args.dynamo.prefill_worker:
        with pytest.raises(ValueError, match=r"Prefill worker configuration is not defined in the test definition."):
            json_gen.gen_prefill_dict()
        return

    prefill = json_gen.gen_prefill_dict()
    assert prefill.get("dynamoNamespace") == system.default_namespace
    assert prefill.get("componentType") == "worker"
    assert prefill.get("replicas") == 1
    assert prefill.get("subComponentType") == "prefill"

    args = ["--model", tdef.cmd_args.dynamo.model, "--is-prefill-worker"]
    for arg, value in dynamo_args_dict(tdef.cmd_args.dynamo.prefill_worker).items():
        args.extend([json_gen._to_dynamo_arg(arg), str(value)])
    if tdef.cmd_args.dynamo.prefill_worker.extra_args:
        args.append(f"{tdef.cmd_args.dynamo.prefill_worker.extra_args}")

    main_container = prefill.get("extraPodSpec", {}).get("mainContainer", {})
    assert main_container.get("image") == tdef.cmd_args.docker_image_url
    assert main_container.get("workingDir") == tdef.cmd_args.dynamo.workspace_path
    assert main_container.get("command") == tdef.cmd_args.dynamo.prefill_cmd.split()
    assert main_container.get("args") == args

    resources = prefill.get("resources", {})
    assert resources.get("limits", {}).get("gpu") == f"{system.gpus_per_node}"


@pytest.mark.parametrize("num_nodes", [1, 2, 4])
def test_gen_prefill_num_nodes(num_nodes: int, json_gen: AIDynamoKubernetesJsonGenStrategy) -> None:
    tdef = cast(AIDynamoTestDefinition, json_gen.test_run.test)
    if not tdef.cmd_args.dynamo.prefill_worker:
        pytest.skip("Prefill worker configuration is not defined in the test definition.")

    tdef.cmd_args.dynamo.prefill_worker.num_nodes = num_nodes

    prefill = json_gen.gen_prefill_dict()

    if num_nodes > 1:
        multinode = prefill.get("multinode", {})
        assert multinode.get("nodeCount") == num_nodes
    else:
        assert "multinode" not in prefill


@pytest.mark.parametrize(
    "arg_name,expected",
    [
        ("nodes", "--nodes"),
        ("num_nodes", "--num-nodes"),
        ("num-nodes", "--num-nodes"),
    ],
)
def test_to_dynamo_arg(json_gen: AIDynamoKubernetesJsonGenStrategy, arg_name: str, expected: str) -> None:
    assert json_gen._to_dynamo_arg(arg_name) == expected


def test_gen_json(json_gen: AIDynamoKubernetesJsonGenStrategy) -> None:
    k8s_system = cast(KubernetesSystem, json_gen.system)
    tdef = cast(AIDynamoTestDefinition, json_gen.test_run.test)
    json_gen.test_run.output_path.mkdir(parents=True, exist_ok=True)

    deployment = json_gen.gen_json()

    assert deployment.get("apiVersion") == "nvidia.com/v1alpha1"
    assert deployment.get("kind") == "DynamoGraphDeployment"
    assert deployment.get("metadata", {}).get("name") == k8s_system.default_namespace

    if tdef.cmd_args.dynamo.prefill_worker:
        assert "prefill" in deployment.get("spec", {}).get("services", {})
    else:
        assert "spec" in deployment
        assert "services" in deployment["spec"]
        assert "prefill" not in deployment["spec"]["services"]

    with open(json_gen.test_run.output_path / json_gen.DEPLOYMENT_FILE_NAME, "r") as f:
        content = yaml.safe_load(f)
        assert content == deployment
