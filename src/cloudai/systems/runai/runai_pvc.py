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

from typing import Any, Dict, List, Optional


class RunAIPVC:
    """Represent a Persistent Volume Claim (PVC) in the RunAI cluster."""

    def __init__(self, raw: Dict[str, Any]) -> None:
        self._parse_meta(raw.get("meta", {}))
        self._parse_spec(raw.get("spec", {}))
        self.resources: List[Dict[str, Any]] = raw.get("clusterInfo", {}).get("resources", [])

    def _parse_meta(self, meta: Dict[str, Any]) -> None:
        self.name: str = meta.get("name", "")
        self.description: str = meta.get("description", "")
        self.scope: str = meta.get("scope", "")
        self.cluster_id: str = meta.get("clusterId", "")
        self.department_id: Optional[str] = meta.get("departmentId")
        self.project_id: Optional[int] = meta.get("projectId")
        self.auto_delete: bool = meta.get("autoDelete", False)
        self.workload_supported_types: Optional[Dict[str, Any]] = meta.get("workloadSupportedTypes")
        self.id: str = meta.get("id", "")
        self.kind: str = meta.get("kind", "")
        self.tenant_id: int = meta.get("tenantId", -1)
        self.created_by: str = meta.get("createdBy", "")
        self.created_at: str = meta.get("createdAt", "")
        self.updated_by: str = meta.get("updatedBy", "")
        self.updated_at: str = meta.get("updatedAt", "")
        self.deleted_at: Optional[str] = meta.get("deletedAt")
        self.deleted_by: Optional[str] = meta.get("deletedBy")
        self.project_name: Optional[str] = meta.get("projectName")
        self.update_count: Optional[int] = meta.get("updateCount")

    def _parse_spec(self, spec: Dict[str, Any]) -> None:
        self.path: str = spec.get("path", "")
        self.existing_pvc: bool = spec.get("existingPvc", False)
        self.claim_name: str = spec.get("claimName", "")
        self.read_only: bool = spec.get("readOnly", False)
        self.ephemeral: bool = spec.get("ephemeral", False)
        claim_info: Dict[str, Any] = spec.get("claimInfo", {})
        self.access_modes: Optional[Dict[str, bool]] = claim_info.get("accessModes")
        self.size: Optional[str] = claim_info.get("size")
        self.storage_class: Optional[str] = claim_info.get("storageClass")
        self.volume_mode: Optional[str] = claim_info.get("volumeMode")

    def __repr__(self) -> str:
        """Return a string representation of the RunAIPVC object."""
        return (
            f"RunAIPVC(name='{self.name}', id='{self.id}', created_by='{self.created_by}', "
            f"scope='{self.scope}', cluster_id='{self.cluster_id}', project_id={self.project_id}, "
            f"size='{self.size}', storage_class='{self.storage_class}', path='{self.path}', "
            f"claim_name='{self.claim_name}')"
        )
