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
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProjectPhase(Enum):
    """Enumeration of possible project phases."""

    READY = "ready"
    CREATING = "creating"


class RunAIProject(BaseModel):
    """Represent a RunAI Project."""

    description: str = Field(default="", alias="description")
    scheduling_rules: Optional[Dict[str, Any]] = Field(default_factory=dict, alias="schedulingRules")
    default_node_pools: Optional[List[str]] = Field(default_factory=list, alias="defaultNodePools")
    node_types: Dict[str, Any] = Field(default_factory=dict, alias="nodeTypes")
    resources: List[Dict[str, Any]] = Field(default_factory=list, alias="resources")
    name: str = Field(default="", alias="name")
    cluster_id: str = Field(default="", alias="clusterId")
    id: str = Field(default="", alias="id")
    parent_id: str = Field(default="", alias="parentId")
    requested_namespace: Optional[str] = Field(default=None, alias="requestedNamespace")
    enforce_runai_scheduler: bool = Field(default=False, alias="enforceRunaiScheduler")
    status: Dict[str, Any] = Field(default_factory=dict, alias="status")
    total_resources: Dict[str, Any] = Field(default_factory=dict, alias="totalResources")
    created_at: str = Field(default="", alias="createdAt")
    updated_at: str = Field(default="", alias="updatedAt")
    created_by: str = Field(default="", alias="createdBy")
    updated_by: str = Field(default="", alias="updatedBy")
    parent: Optional[Dict[str, Any]] = Field(default=None, alias="parent")
    effective: Dict[str, Any] = Field(default_factory=dict, alias="effective")
    overtime_data: Dict[str, Any] = Field(default_factory=dict, alias="overtimeData")
    range_24h_data: Optional[Dict[str, Any]] = Field(default=None, alias="range24hData")
    range_7d_data: Optional[Dict[str, Any]] = Field(default=None, alias="range7dData")
    range_30d_data: Optional[Dict[str, Any]] = Field(default=None, alias="range30dData")

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    def __repr__(self) -> str:
        """Prettify project output."""
        phase: str = self.status.get("phase", "")
        return (
            f"RunAIProject(name={self.name!r}, id={self.id!r}, cluster_id={self.cluster_id!r}, "
            f"created_by={self.created_by!r}, phase={phase!r})"
        )
