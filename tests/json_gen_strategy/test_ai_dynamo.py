from typing import cast

import pytest

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


@pytest.fixture
def dynamo() -> AIDynamoTestDefinition:
    return AIDynamoTestDefinition(
        name="test_dynamo",
        description="Test AI Dynamo workload",
        test_template_name="AIDynamo",
        cmd_args=AIDynamoCmdArgs(
            docker_image_url="nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1.post1",
            dynamo=AIDynamoArgs(
                model="model",
                backend="vllm",
                workspace_path="/workspace/examples/backends/vllm",
                prefill_worker=PrefillWorkerArgs(num_nodes=2),
                decode_worker=DecodeWorkerArgs(num_nodes=2),
            ),
            genai_perf=GenAIPerfArgs(),
        ),
    )


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


def test_gen_decode(json_gen: AIDynamoKubernetesJsonGenStrategy) -> None:
    system = cast(KubernetesSystem, json_gen.system)
    tdef = cast(AIDynamoTestDefinition, json_gen.test_run.test)

    decode = json_gen.gen_decode_dict()
    assert decode.get("dynamoNamespace") is system.default_namespace
    assert decode.get("componentType") == "worker"
    assert decode.get("replicas") == 1

    main_container = decode.get("extraPodSpec", {}).get("mainContainer", {})
    assert main_container.get("image") == tdef.cmd_args.docker_image_url
    assert main_container.get("workingDir") == tdef.cmd_args.dynamo.workspace_path
    assert main_container.get("command") == tdef.cmd_args.dynamo.decode_cmd.split()
    assert main_container.get("args") == ["--model", tdef.cmd_args.dynamo.model]

    resources = decode.get("resources", {})
    assert resources.get("limits", {}).get("gpu") == f"{system.gpus_per_node}"


def test_gen_json(json_gen: AIDynamoKubernetesJsonGenStrategy) -> None:
    k8s_system = cast(KubernetesSystem, json_gen.system)
    json_gen._setup_genai = lambda td: None
    deployment = json_gen.gen_json()
    assert deployment.get("apiVersion") == "nvidia.com/v1alpha1"
    assert deployment.get("kind") == "DynamoGraphDeployment"
    assert deployment.get("metadata", {}).get("name") == k8s_system.default_namespace
