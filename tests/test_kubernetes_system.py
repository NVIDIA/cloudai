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
from unittest.mock import MagicMock, patch

import pytest
from cloudai.systems.kubernetes.kubernetes_system import KubernetesSystem
from kubernetes import client


@pytest.fixture
def k8s_system():
    """Fixture to create a KubernetesSystem instance."""
    with patch("kubernetes.config.load_kube_config"), patch("pathlib.Path.exists", return_value=True):
        k8s_system = KubernetesSystem(
            name="test-system",
            install_path=Path("/fake/install/path"),
            output_path=Path("/fake/output/path"),
            kube_config_path=Path("/fake/kube/config"),
            default_namespace="default",
            default_image="test-image",
        )
        k8s_system._core_v1 = MagicMock(client.CoreV1Api)
        k8s_system._batch_v1 = MagicMock(client.BatchV1Api)
        k8s_system._custom_objects_api = MagicMock(client.CustomObjectsApi)
        yield k8s_system


def test_initialization(k8s_system):
    """Test that all attributes are properly initialized."""
    assert k8s_system.name == "test-system"
    assert k8s_system.install_path == Path("/fake/install/path")
    assert k8s_system.output_path == Path("/fake/output/path")
    assert k8s_system.kube_config_path == Path("/fake/kube/config")
    assert k8s_system.default_namespace == "default"
    assert k8s_system.default_image == "test-image"
    assert k8s_system.scheduler == "kubernetes"
    assert k8s_system.global_env_vars == {}
    assert k8s_system.monitor_interval == 1
