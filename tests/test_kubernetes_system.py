# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from cloudai.systems.kubernetes.kubernetes_system import KubernetesSystem
from kubernetes import client


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
    k8s_system = KubernetesSystem(
        name="test-system",
        install_path=Path("/fake/install/path"),
        output_path=Path("/fake/output/path"),
        kube_config_path=kube_config_tempfile,
        default_namespace="default",
        default_image="test-image",
    )
    k8s_system.model_post_init(None)

    validated_system = KubernetesSystem.model_validate(k8s_system.model_dump())

    yield validated_system


def test_initialization(k8s_system):
    """Test that all attributes and Kubernetes API clients are properly initialized."""
    assert k8s_system.name == "test-system"
    assert k8s_system.install_path == Path("/fake/install/path")
    assert k8s_system.output_path == Path("/fake/output/path")
    assert k8s_system.kube_config_path.exists()
    assert k8s_system.default_namespace == "default"
    assert k8s_system.default_image == "test-image"
    assert k8s_system.scheduler == "kubernetes"
    assert k8s_system.global_env_vars == {}
    assert k8s_system.monitor_interval == 1

    assert isinstance(k8s_system.core_v1, client.CoreV1Api)
    assert isinstance(k8s_system.batch_v1, client.BatchV1Api)
    assert isinstance(k8s_system.custom_objects_api, client.CustomObjectsApi)
