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

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClusterState(Enum):
    """Enumeration of possible cluster states."""

    WAITING_TO_CONNECT = "WaitingToConnect"
    CONNECTED = "Connected"
    DISCONNECTED = "Disconnected"
    MISSING_PREREQUISITES = "MissingPrerequisites"
    SERVICE_ISSUES = "ServiceIssues"
    UNKNOWN = "Unknown"

    @classmethod
    def from_str(cls, value: str) -> ClusterState:
        return cls(cls._value2member_map_.get(value, cls.UNKNOWN))


class Status(BaseModel):
    """Represent the status of a RunAI cluster."""

    state: ClusterState = Field(default=ClusterState.UNKNOWN)
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    operands: Dict[str, Any] = Field(default_factory=dict)
    platform: Optional[Dict[str, Any]] = Field(default=None)
    config: Optional[Dict[str, Any]] = Field(default=None)
    dependencies: Dict[str, Any] = Field(default_factory=dict)


class RunAICluster(BaseModel):
    """Represent a RunAI cluster with its associated data and operations."""

    uuid: str = Field(default="", alias="uuid")
    tenant_id: int = Field(default=0, alias="tenantId")
    name: str = Field(default="", alias="name")
    created_at: str = Field(default="", alias="createdAt")
    domain: Optional[str] = Field(default=None, alias="domain")
    version: Optional[str] = Field(default=None, alias="version")
    updated_at: Optional[str] = Field(default=None, alias="updatedAt")
    deleted_at: Optional[str] = Field(default=None, alias="deletedAt")
    last_liveness: Optional[str] = Field(default=None, alias="lastLiveness")
    delete_requested_at: Optional[str] = Field(default=None, alias="deleteRequestedAt")

    status: Status = Field(default_factory=Status, alias="status")

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    def is_connected(self) -> bool:
        return self.status.state == ClusterState.CONNECTED

    def get_kubernetes_version(self) -> Optional[str]:
        if not self.status.platform:
            return None
        return self.status.platform.get("kubeVersion")

    def __repr__(self) -> str:
        """Return a string representation of the RunAICluster object."""
        return (
            f"RunAICluster(name={self.name!r}, uuid={self.uuid!r}, tenant_id={self.tenant_id}, "
            f"state={self.status.state.value!r}, created_at={self.created_at!r}, version={self.version!r}, "
            f"domain={self.domain!r})"
        )
