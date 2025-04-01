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
from typing import Any, Dict, List, Optional


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


class RunAICluster:
    """Represent a RunAI cluster with its associated data and operations."""

    def __init__(self, data: Dict[str, Any]) -> None:
        """Initialize a RunAICluster with data from a dictionary."""
        self.uuid: str = data.get("uuid", "")
        self.tenant_id: int = data.get("tenantId", 0)
        self.name: str = data.get("name", "")
        self.created_at: str = data.get("createdAt", "")
        self.domain: Optional[str] = data.get("domain")
        self.version: Optional[str] = data.get("version")
        self.updated_at: Optional[str] = data.get("updatedAt")
        self.deleted_at: Optional[str] = data.get("deletedAt")
        self.last_liveness: Optional[str] = data.get("lastLiveness")
        self.delete_requested_at: Optional[str] = data.get("deleteRequestedAt")

        status: Dict[str, Any] = data.get("status", {})
        self.state: ClusterState = ClusterState.from_str(status.get("state", "Unknown"))
        self.conditions: List[Dict[str, Any]] = status.get("conditions", [])
        self.operands: Dict[str, Any] = status.get("operands", {})
        self.platform: Optional[Dict[str, Any]] = status.get("platform")
        self.config: Optional[Dict[str, Any]] = status.get("config")
        self.dependencies: Dict[str, Any] = status.get("dependencies", {})

    def is_connected(self) -> bool:
        return self.state == ClusterState.CONNECTED

    def get_kubernetes_version(self) -> Optional[str]:
        if not self.platform:
            return None
        return self.platform.get("kubeVersion")

    def __repr__(self) -> str:
        """Return a string representation of the RunAICluster object."""
        return (
            f"RunAICluster(name='{self.name}', uuid='{self.uuid}', tenant_id={self.tenant_id}, "
            f"state='{self.state.value}', created_at='{self.created_at}', version='{self.version}', "
            f"domain='{self.domain}')"
        )
