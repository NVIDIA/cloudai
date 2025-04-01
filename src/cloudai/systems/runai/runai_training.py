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

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from dateutil import parser as dateutil_parser


class WorkloadPhase(str, Enum):
    """The phase of the workload."""

    RUNNING = "Running"
    STOPPED = "Stopped"
    DELETED = "Deleted"


class ActualPhase(str, Enum):
    """The actual phase of the workload."""

    CREATING = "Creating"
    INITIALIZING = "Initializing"
    RESUMING = "Resuming"
    PENDING = "Pending"
    DELETING = "Deleting"
    RUNNING = "Running"
    UPDATING = "Updating"
    STOPPED = "Stopped"
    STOPPING = "Stopping"
    DEGRADED = "Degraded"
    FAILED = "Failed"
    COMPLETED = "Completed"
    TERMINATING = "Terminating"
    UNKNOWN = "Unknown"


@dataclass
class Compute:
    """The compute resources requested for the workload."""

    gpu_devices_request: Optional[int] = None
    gpu_request_type: Optional[str] = None
    gpu_portion_request: Optional[float] = None
    gpu_portion_limit: Optional[float] = None
    gpu_memory_request: Optional[str] = None
    gpu_memory_limit: Optional[str] = None
    mig_profile: Optional[str] = None
    cpu_core_request: Optional[float] = None
    cpu_core_limit: Optional[float] = None
    cpu_memory_request: Optional[str] = None
    cpu_memory_limit: Optional[str] = None
    large_shm_request: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Compute":
        return cls(
            gpu_devices_request=data.get("gpuDevicesRequest"),
            gpu_request_type=data.get("gpuRequestType"),
            gpu_portion_request=data.get("gpuPortionRequest"),
            gpu_portion_limit=data.get("gpuPortionLimit"),
            gpu_memory_request=data.get("gpuMemoryRequest"),
            gpu_memory_limit=data.get("gpuMemoryLimit"),
            mig_profile=data.get("migProfile"),
            cpu_core_request=data.get("cpuCoreRequest"),
            cpu_core_limit=data.get("cpuCoreLimit"),
            cpu_memory_request=data.get("cpuMemoryRequest"),
            cpu_memory_limit=data.get("cpuMemoryLimit"),
            large_shm_request=data.get("largeShmRequest"),
        )


@dataclass
class StoragePVC:
    """The storage resources requested for the workload."""

    claim_name: str
    path: str
    existing_pvc: Optional[bool] = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoragePVC":
        return cls(
            claim_name=data.get("claimName", ""),
            path=data.get("path", ""),
            existing_pvc=data.get("existingPvc", False),
        )


@dataclass
class Storage:
    """The storage resources requested for the workload."""

    pvc: List[StoragePVC] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Storage":
        pvc_data = data.get("pvc", [])
        return cls(pvc=[StoragePVC.from_dict(p) for p in pvc_data])


@dataclass
class Security:
    """The security resources requested for the workload."""

    run_as_uid: Optional[int] = None
    run_as_gid: Optional[int] = None
    allow_privilege_escalation: Optional[bool] = None
    host_ipc: Optional[bool] = None
    host_network: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Security":
        return cls(
            run_as_uid=data.get("runAsUid"),
            run_as_gid=data.get("runAsGid"),
            allow_privilege_escalation=data.get("allowPrivilegeEscalation"),
            host_ipc=data.get("hostIpc"),
            host_network=data.get("hostNetwork"),
        )


@dataclass
class TrainingSpec:
    """The specification of the workload."""

    command: Optional[str] = None
    args: Optional[str] = None
    image: Optional[str] = None
    restart_policy: Optional[str] = None
    node_pools: List[str] = field(default_factory=list)
    compute: Optional[Compute] = None
    storage: Optional[Storage] = None
    security: Optional[Security] = None
    completions: Optional[int] = None
    parallelism: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingSpec":
        return cls(
            command=data.get("command"),
            args=data.get("args"),
            image=data.get("image"),
            restart_policy=data.get("restartPolicy"),
            node_pools=data.get("nodePools", []),
            compute=Compute.from_dict(data.get("compute", {})),
            storage=Storage.from_dict(data.get("storage", {})),
            security=Security.from_dict(data.get("security", {})),
            completions=data.get("completions"),
            parallelism=data.get("parallelism"),
        )


@dataclass
class RunAITraining:
    """The training workload."""

    name: str
    requested_name: str
    workload_id: str
    project_id: str
    cluster_id: str
    created_by: str
    created_at: datetime
    deleted_at: Optional[datetime]
    desired_phase: WorkloadPhase
    actual_phase: ActualPhase
    spec: TrainingSpec

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunAITraining":
        return cls(
            name=data["name"],
            requested_name=data["requestedName"],
            workload_id=data["workloadId"],
            project_id=str(data["projectId"]),
            cluster_id=data["clusterId"],
            created_by=data["createdBy"],
            created_at=dateutil_parser.parse(data["createdAt"]),
            deleted_at=dateutil_parser.parse(data["deletedAt"]) if data.get("deletedAt") else None,
            desired_phase=WorkloadPhase(data["desiredPhase"]),
            actual_phase=ActualPhase(data["actualPhase"]),
            spec=TrainingSpec.from_dict(data.get("spec", {})),
        )
