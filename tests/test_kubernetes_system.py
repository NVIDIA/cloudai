# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
from kubernetes import client

from cloudai.systems.kubernetes.kubernetes_system import KubernetesSystem


@pytest.fixture
def kube_config_tempfile():
    """Fixture to create a kube config file in $HOME/.kube/config with reasonable content."""
    kube_config_content = """
    apiVersion: v1
    kind: Config
    clusters:
    - cluster:
        server: https://127.0.0.1:6443
      name: local-cluster
    contexts:
    - context:
        cluster: local-cluster
        user: local-user
      name: local-context
    current-context: local-context
    users:
    - name: local-user
      user:
        token: fake-token
    """

    home_dir = Path.home()
    kube_config_dir = home_dir / ".kube"
    kube_config_path = kube_config_dir / "config"

    kube_config_dir.mkdir(parents=True, exist_ok=True)

    with kube_config_path.open("w") as config_file:
        config_file.write(kube_config_content)

    yield kube_config_path


@pytest.fixture
def k8s_system(kube_config_tempfile):
    """Fixture to create a KubernetesSystem instance with a valid kube config."""
    system_data = {
        "name": "test-system",
        "install_path": "/fake/install/path",
        "output_path": "/fake/output/path",
        "kube_config_path": kube_config_tempfile,
        "default_namespace": "default",
        "scheduler": "kubernetes",
        "global_env_vars": {},
        "monitor_interval": 1,
    }

    k8s_system = KubernetesSystem(**system_data)
    k8s_system.model_post_init(None)

    validated_system = KubernetesSystem.model_validate(system_data)

    yield validated_system


def test_initialization(k8s_system):
    """Test that all attributes and Kubernetes API clients are properly initialized."""
    assert k8s_system.name == "test-system"
    assert k8s_system.install_path == Path("/fake/install/path")
    assert k8s_system.output_path == Path("/fake/output/path")
    assert k8s_system.kube_config_path.exists()
    assert k8s_system.default_namespace == "default"
    assert k8s_system.scheduler == "kubernetes"
    assert k8s_system.global_env_vars == {}
    assert k8s_system.monitor_interval == 1

    assert isinstance(k8s_system.core_v1, client.CoreV1Api)
    assert isinstance(k8s_system.batch_v1, client.BatchV1Api)
    assert isinstance(k8s_system.custom_objects_api, client.CustomObjectsApi)


@pytest.mark.parametrize(
    "pod_output,expected_result",
    [
        # Test case 1: No vLLM pods
        ("NAME  READY  STATUS\npod1  1/1    Running", False),
        # Test case 2: vLLM pod running and ready
        ("NAME                READY   STATUS\nvllm-v1-agg-pod  1/1     Running", True),
        # Test case 3: vLLM pod not ready
        ("NAME                READY   STATUS\nvllm-v1-agg-pod  0/1     Running", False),
        # Test case 4: vLLM pod terminating
        ("NAME                READY   STATUS\nvllm-v1-agg-pod  1/1     Terminating", False),
        # Test case 5: Multiple pods, vLLM pod ready
        (
            "NAME                READY   STATUS\n"
            "pod1                1/1     Running\n"
            "vllm-v1-agg-pod    1/1     Running",
            True,
        ),
        # Test case 6: Multiple vLLM pods, all must be ready
        (
            "NAME                READY   STATUS\n"
            "vllm-v1-agg-pod1   1/1     Running\n"
            "vllm-v1-agg-pod2   0/1     Running",
            False,
        ),
    ],
)
def test_are_vllm_pods_ready(k8s_system: KubernetesSystem, pod_output: str, expected_result: bool) -> None:
    """Test the are_vllm_pods_ready method with various pod states."""
    with patch("subprocess.run") as mock_run:
        mock_process = MagicMock()
        mock_process.stdout = pod_output
        mock_process.returncode = 0
        mock_run.return_value = mock_process

        assert k8s_system.are_vllm_pods_ready() == expected_result


@pytest.mark.parametrize(
    "conditions,expected_result",
    [
        # Test case 1: Empty conditions list
        ([], True),
        # Test case 2: Ready condition is True
        ([{"type": "Ready", "status": "True"}], True),
        # Test case 3: Ready condition is False
        ([{"type": "Ready", "status": "False"}], True),
        # Test case 4: Failed condition is True
        ([{"type": "Failed", "status": "True"}], False),
        # Test case 5: Failed condition is False
        ([{"type": "Failed", "status": "False"}], True),
        # Test case 6: Multiple conditions with Ready True
        (
            [
                {"type": "Created", "status": "True"},
                {"type": "Ready", "status": "True"},
                {"type": "Running", "status": "True"},
            ],
            True,
        ),
        # Test case 7: Multiple conditions with Failed True
        (
            [
                {"type": "Created", "status": "True"},
                {"type": "Failed", "status": "True"},
                {"type": "Running", "status": "False"},
            ],
            False,
        ),
        # Test case 8: Other conditions only
        (
            [
                {"type": "Created", "status": "True"},
                {"type": "Running", "status": "True"},
            ],
            True,
        ),
    ],
)
def test_check_deployment_conditions(
    k8s_system: KubernetesSystem, conditions: List[Dict[str, str]], expected_result: bool
) -> None:
    assert k8s_system._check_deployment_conditions(conditions) == expected_result
