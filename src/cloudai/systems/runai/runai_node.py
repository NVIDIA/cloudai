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

from enum import Enum
from typing import Any, Dict, List, Optional


class NodeStatus(Enum):
    """Enumeration of possible node statuses."""

    READY = "Ready"
    NOT_READY = "NotReady"
    UNKNOWN = "Unknown"

    @classmethod
    def from_str(cls, value: str) -> "NodeStatus":
        try:
            return cls(value)
        except ValueError:
            return cls.UNKNOWN


class RunAINode:
    """Represent a node in the RunAI cluster."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.status: NodeStatus = NodeStatus.from_str(data.get("status", "Unknown"))
        self.conditions: List[Dict[str, Any]] = data.get("conditions", [])
        self.taints: List[Dict[str, Any]] = data.get("taints", [])
        self.node_pool: str = data.get("nodePool", "")
        self.created_at: str = data.get("createdAt", "")

        gpu_info: Optional[Dict[str, Any]] = data.get("gpuInfo")
        self.gpu_type: Optional[str] = gpu_info.get("gpuType") if gpu_info else None
        self.gpu_count: Optional[int] = gpu_info.get("gpuCount") if gpu_info else None
        self.nvlink_domain_uid: Optional[str] = gpu_info.get("nvLinkDomainUid") if gpu_info else None
        self.nvlink_clique_id: Optional[str] = gpu_info.get("nvLinkCliqueId") if gpu_info else None

        self.name: str = data.get("name", "")
        self.id: str = data.get("id", "")
        self.cluster_uuid: str = data.get("clusterUuid", "")
        self.updated_at: str = data.get("updatedAt", "")

    def __repr__(self) -> str:
        """Return a string representation of the RunAINode object."""
        return (
            f"RunAINode(name='{self.name}', id='{self.id}', cluster_uuid='{self.cluster_uuid}', "
            f"node_pool='{self.node_pool}', gpu_type='{self.gpu_type}', gpu_count={self.gpu_count}, "
            f"nvlink_domain_uid='{self.nvlink_domain_uid}', nvlink_clique_id='{self.nvlink_clique_id}', "
            f"status='{self.status.value}')"
        )
