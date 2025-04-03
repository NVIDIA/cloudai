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

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


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


class Compute(BaseModel):
    """Represent the compute resources requested for the workload."""

    gpu_devices_request: Optional[int] = Field(default=None, alias="gpuDevicesRequest")
    gpu_request_type: Optional[str] = Field(default=None, alias="gpuRequestType")
    gpu_portion_request: Optional[float] = Field(default=None, alias="gpuPortionRequest")
    gpu_portion_limit: Optional[float] = Field(default=None, alias="gpuPortionLimit")
    gpu_memory_request: Optional[str] = Field(default=None, alias="gpuMemoryRequest")
    gpu_memory_limit: Optional[str] = Field(default=None, alias="gpuMemoryLimit")
    mig_profile: Optional[str] = Field(default=None, alias="migProfile")
    cpu_core_request: Optional[float] = Field(default=None, alias="cpuCoreRequest")
    cpu_core_limit: Optional[float] = Field(default=None, alias="cpuCoreLimit")
    cpu_memory_request: Optional[str] = Field(default=None, alias="cpuMemoryRequest")
    cpu_memory_limit: Optional[str] = Field(default=None, alias="cpuMemoryLimit")
    large_shm_request: Optional[bool] = Field(default=None, alias="largeShmRequest")


class StoragePVC(BaseModel):
    """Represent the storage resources requested for the workload."""

    claim_name: str = Field(default="", alias="claimName")
    path: str = Field(default="", alias="path")
    existing_pvc: Optional[bool] = Field(default=False, alias="existingPvc")


class Storage(BaseModel):
    """Represent the storage configuration for the workload."""

    pvc: List[StoragePVC] = Field(default_factory=list, alias="pvc")


class Security(BaseModel):
    """Represent the security settings for the workload."""

    run_as_uid: Optional[int] = Field(default=None, alias="runAsUid")
    run_as_gid: Optional[int] = Field(default=None, alias="runAsGid")
    allow_privilege_escalation: Optional[bool] = Field(default=None, alias="allowPrivilegeEscalation")
    host_ipc: Optional[bool] = Field(default=None, alias="hostIpc")
    host_network: Optional[bool] = Field(default=None, alias="hostNetwork")


class TrainingSpec(BaseModel):
    """Represent the specification of the training workload."""

    command: Optional[str] = Field(default=None, alias="command")
    args: Optional[str] = Field(default=None, alias="args")
    image: Optional[str] = Field(default=None, alias="image")
    restart_policy: Optional[str] = Field(default=None, alias="restartPolicy")
    node_pools: List[str] = Field(default_factory=list, alias="nodePools")
    compute: Optional[Compute] = Field(default=None, alias="compute")
    storage: Optional[Storage] = Field(default=None, alias="storage")
    security: Optional[Security] = Field(default=None, alias="security")
    completions: Optional[int] = Field(default=None, alias="completions")
    parallelism: Optional[int] = Field(default=None, alias="parallelism")


class RunAITraining(BaseModel):
    """Represent a training workload in the RunAI system."""

    name: str = Field(alias="name")
    requested_name: str = Field(alias="requestedName")
    workload_id: str = Field(alias="workloadId")
    project_id: str = Field(alias="projectId")
    cluster_id: str = Field(alias="clusterId")
    created_by: str = Field(alias="createdBy")
    created_at: datetime = Field(alias="createdAt")
    deleted_at: Optional[datetime] = Field(default=None, alias="deletedAt")
    desired_phase: WorkloadPhase = Field(alias="desiredPhase")
    actual_phase: ActualPhase = Field(alias="actualPhase")
    spec: TrainingSpec = Field(alias="spec")

    class Config:
        """Pydantic configuration for RunAITraining."""

        populate_by_name = True
        populate_by_alias = True
        validate_by_alias = True
