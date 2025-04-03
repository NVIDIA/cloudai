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

from typing import Any, Dict

import pytest

from cloudai.systems.runai.runai_cluster import ClusterState, RunAICluster


@pytest.fixture
def cluster_data() -> Dict[str, Any]:
    return {
        "createdAt": "2025-01-01T00:00:00Z",
        "domain": "https://dummy-cluster.example.com",
        "lastLiveness": "2025-01-02T00:00:00Z",
        "name": "dummy-cluster-name",
        "status": {
            "state": "Disconnected",
            "conditions": [],
            "platform": {"kubeVersion": "v0.0.0-dummy", "type": "dummy-platform"},
            "config": {},
            "dependencies": {},
            "operands": {},
        },
        "tenantId": 9999,
        "updatedAt": "2025-01-01T00:00:00Z",
        "uuid": "00000000-0000-0000-0000-000000000000",
        "version": "0.0.0",
    }


def test_cluster_state_enum() -> None:
    assert ClusterState.from_str("Connected") == ClusterState.CONNECTED
    assert ClusterState.from_str("Disconnected") == ClusterState.DISCONNECTED
    assert ClusterState.from_str("UnknownState") == ClusterState.UNKNOWN


def test_cluster_initialization(cluster_data: Dict[str, Any]) -> None:
    cluster = RunAICluster(**cluster_data)
    assert cluster.name == cluster_data["name"]
    assert cluster.uuid == cluster_data["uuid"]
    assert cluster.tenant_id == cluster_data["tenantId"]
    assert cluster.domain == cluster_data["domain"]
    assert cluster.version == cluster_data["version"]
    assert cluster.created_at == cluster_data["createdAt"]
    assert cluster.updated_at == cluster_data["updatedAt"]
    assert cluster.last_liveness == cluster_data["lastLiveness"]
    assert cluster.state == ClusterState.DISCONNECTED


def test_cluster_state_disconnected(cluster_data: Dict[str, Any]) -> None:
    cluster = RunAICluster(**cluster_data)
    assert not cluster.is_connected()


def test_cluster_state_connected(cluster_data: Dict[str, Any]) -> None:
    cluster_data["status"]["state"] = "Connected"
    cluster = RunAICluster(**cluster_data)
    assert cluster.is_connected()


def test_cluster_kubernetes_version(cluster_data: Dict[str, Any]) -> None:
    cluster = RunAICluster(**cluster_data)
    assert cluster.get_kubernetes_version() == "v0.0.0-dummy"


def test_repr(cluster_data: Dict[str, Any]) -> None:
    cluster = RunAICluster(**cluster_data)
    output = repr(cluster)
    assert isinstance(output, str)
    assert cluster.name is not None and cluster.name in output
    assert cluster.uuid is not None and cluster.uuid in output
    assert cluster.version is not None and cluster.version in output
