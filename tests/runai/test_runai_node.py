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

import pytest

from cloudai.systems.runai.runai_node import NodeStatus, RunAINode


@pytest.mark.parametrize(
    "input_status, expected_status",
    [
        ("Ready", NodeStatus.READY),
        ("NotReady", NodeStatus.NOT_READY),
        ("Unknown", NodeStatus.UNKNOWN),
        ("GarbageStatus", NodeStatus.UNKNOWN),
        ("", NodeStatus.UNKNOWN),
        (None, NodeStatus.UNKNOWN),
    ],
)
def test_node_status_parsing(input_status, expected_status) -> None:
    status = NodeStatus.from_str(input_status if input_status is not None else "")
    assert status == expected_status


@pytest.fixture
def sample_node_data() -> dict:
    return {
        "status": "Ready",
        "conditions": [{"type": "MemoryPressure", "reason": "KubeletHasSufficientMemory"}],
        "taints": [{"key": "dedicated", "value": "gpu", "effect": "NoSchedule"}],
        "nodePool": "gpu-pool",
        "createdAt": "2025-01-01T00:00:00Z",
        "gpuInfo": {
            "gpuType": "NVIDIA-A100",
            "gpuCount": 8,
            "nvLinkDomainUid": "domain-1234",
            "nvLinkCliqueId": "clique-5678",
        },
        "name": "node-1",
        "id": "node-uuid",
        "clusterUuid": "cluster-uuid",
        "updatedAt": "2025-01-02T00:00:00Z",
    }


def test_runai_node_properties(sample_node_data) -> None:
    node = RunAINode(sample_node_data)

    assert node.status == NodeStatus.READY
    assert node.conditions == sample_node_data["conditions"]
    assert node.taints == sample_node_data["taints"]
    assert node.node_pool == "gpu-pool"
    assert node.created_at == "2025-01-01T00:00:00Z"
    assert node.gpu_type == "NVIDIA-A100"
    assert node.gpu_count == 8
    assert node.nvlink_domain_uid == "domain-1234"
    assert node.nvlink_clique_id == "clique-5678"
    assert node.name == "node-1"
    assert node.id == "node-uuid"
    assert node.cluster_uuid == "cluster-uuid"
    assert node.updated_at == "2025-01-02T00:00:00Z"


def test_runai_node_repr(sample_node_data) -> None:
    node = RunAINode(sample_node_data)
    text = repr(node)
    assert "RunAINode(name='node-1'" in text
    assert "gpu_count=8" in text
    assert "status='Ready'" in text
