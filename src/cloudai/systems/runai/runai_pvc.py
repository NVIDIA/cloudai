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

from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PVCMeta(BaseModel):
    """Represent the metadata of a Persistent Volume Claim (PVC)."""

    id: str = Field(default="", alias="id")
    name: str = Field(default="", alias="name")
    kind: str = Field(default="", alias="kind")
    scope: str = Field(default="", alias="scope")
    cluster_id: str = Field(default="", alias="clusterId")
    department_id: Optional[str] = Field(default=None, alias="departmentId")
    tenant_id: int = Field(default=-1, alias="tenantId")
    created_by: str = Field(default="", alias="createdBy")
    created_at: str = Field(default="", alias="createdAt")
    updated_by: str = Field(default="", alias="updatedBy")
    updated_at: str = Field(default="", alias="updatedAt")
    deleted_at: Optional[str] = Field(default=None, alias="deletedAt")
    deleted_by: Optional[str] = Field(default=None, alias="deletedBy")
    description: str = Field(default="", alias="description")
    auto_delete: bool = Field(default=False, alias="autoDelete")
    project_id: Optional[int] = Field(default=None, alias="projectId")
    workload_supported_types: Optional[Dict[str, Any]] = Field(default=None, alias="workloadSupportedTypes")
    project_name: Optional[str] = Field(default=None, alias="projectName")
    update_count: Optional[int] = Field(default=None, alias="updateCount")


class PVCCLAimInfo(BaseModel):
    """Represent the claim information of a Persistent Volume Claim (PVC)."""

    access_modes: Optional[Dict[str, bool]] = Field(default=None, alias="accessModes")
    size: Optional[str] = Field(default=None, alias="size")
    storage_class: Optional[str] = Field(default=None, alias="storageClass")
    volume_mode: Optional[str] = Field(default=None, alias="volumeMode")


class PVCSpec(BaseModel):
    """Represent the specification of a Persistent Volume Claim (PVC)."""

    claim_name: str = Field(default="", alias="claimName")
    path: str = Field(default="", alias="path")
    read_only: bool = Field(default=False, alias="readOnly")
    ephemeral: bool = Field(default=False, alias="ephemeral")
    existing_pvc: bool = Field(default=False, alias="existingPvc")
    claim_info: Optional[PVCCLAimInfo] = Field(default=None, alias="claimInfo")


class PVCClusterInfo(BaseModel):
    """Represent the cluster information of a Persistent Volume Claim (PVC)."""

    resources: List[Dict[str, Any]] = Field(default_factory=list, alias="resources")


class RunAIPVC(BaseModel):
    """Represent a Persistent Volume Claim (PVC) in the RunAI cluster."""

    meta: PVCMeta = Field(alias="meta")
    spec: PVCSpec = Field(alias="spec")
    cluster_info: PVCClusterInfo = Field(alias="clusterInfo")
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    def __repr__(self) -> str:
        """Return a string representation of the RunAIPVC object."""
        return (
            f"RunAIPVC(name={self.meta.name!r}, id={self.meta.id!r}, "
            f"created_by={self.meta.created_by!r}, scope={self.meta.scope!r}, "
            f"cluster_id={self.meta.cluster_id!r}, project_id={self.meta.project_id}, "
            f"size={(self.spec.claim_info.size if self.spec.claim_info else None)!r}, "
            f"storage_class={(self.spec.claim_info.storage_class if self.spec.claim_info else None)!r}, "
            f"path={self.spec.path!r}, claim_name={self.spec.claim_name!r})"
        )
