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


class ProjectPhase(Enum):
    """Enumeration of possible project phases."""

    READY = "ready"
    CREATING = "creating"


class RunAIProject:
    """Represent a RunAI Project."""

    def __init__(self, data: Dict[str, Any]) -> None:
        """Initialize project from a project dictionary."""
        self.description: str = data.get("description", "")
        self.scheduling_rules: Optional[Dict[str, Any]] = data.get("schedulingRules", {})
        self.default_node_pools: Optional[List[str]] = data.get("defaultNodePools", [])
        self.node_types: Dict[str, Any] = data.get("nodeTypes", {})
        self.resources: List[Dict[str, Any]] = data.get("resources", [])
        self.name: str = data.get("name", "")
        self.cluster_id: str = data.get("clusterId", "")
        self.id: str = data.get("id", "")
        self.parent_id: str = data.get("parentId", "")
        self.requested_namespace: Optional[str] = data.get("requestedNamespace")
        self.enforce_runai_scheduler: bool = data.get("enforceRunaiScheduler", False)
        self.status: Dict[str, Any] = data.get("status", {})
        self.total_resources: Dict[str, Any] = data.get("totalResources", {})
        self.created_at: str = data.get("createdAt", "")
        self.updated_at: str = data.get("updatedAt", "")
        self.created_by: str = data.get("createdBy", "")
        self.updated_by: str = data.get("updatedBy", "")
        self.parent: Optional[Dict[str, Any]] = data.get("parent")
        self.effective: Dict[str, Any] = data.get("effective", {})
        self.overtime_data: Dict[str, Any] = data.get("overtimeData", {})
        self.range_24h_data: Optional[Dict[str, Any]] = self.overtime_data.get("range24hData")
        self.range_7d_data: Optional[Dict[str, Any]] = self.overtime_data.get("range7dData")
        self.range_30d_data: Optional[Dict[str, Any]] = self.overtime_data.get("range30dData")

    def __repr__(self) -> str:
        """Prettify project output."""
        phase: str = self.status.get("phase", "")
        return (
            f"RunAIProject(name='{self.name}', id='{self.id}', cluster_id='{self.cluster_id}', "
            f"created_by='{self.created_by}', phase='{phase}')"
        )
