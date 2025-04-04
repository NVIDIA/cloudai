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

from pydantic import BaseModel, ConfigDict, Field, root_validator


class NodeStatus(Enum):
    """Enumeration of possible node statuses."""

    READY = "Ready"
    NOT_READY = "NotReady"
    UNKNOWN = "Unknown"

    @classmethod
    def from_str(cls, value: str) -> "NodeStatus":
        return NodeStatus(cls._value2member_map_.get(value, cls.UNKNOWN))


class RunAINode(BaseModel):
    """Represent a node in the RunAI cluster."""

    status: NodeStatus = Field(default=NodeStatus.UNKNOWN, alias="status")
    conditions: List[Dict[str, Any]] = Field(default_factory=list, alias="conditions")
    taints: List[Dict[str, Any]] = Field(default_factory=list, alias="taints")
    node_pool: str = Field(default="", alias="nodePool")
    created_at: str = Field(default="", alias="createdAt")

    gpu_type: Optional[str] = Field(default=None, alias="gpuType")
    gpu_count: Optional[int] = Field(default=None, alias="gpuCount")
    nvlink_domain_uid: Optional[str] = Field(default=None, alias="nvLinkDomainUid")
    nvlink_clique_id: Optional[str] = Field(default=None, alias="nvLinkCliqueId")

    name: str = Field(default="", alias="name")
    id: str = Field(default="", alias="id")
    cluster_uuid: str = Field(default="", alias="clusterUuid")
    updated_at: str = Field(default="", alias="updatedAt")

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    @root_validator(pre=True)
    def extract_gpu_info(cls, values):
        gpu_info = values.pop("gpuInfo", None)
        if gpu_info and isinstance(gpu_info, dict):
            values["gpuType"] = gpu_info.get("gpuType")
            values["gpuCount"] = gpu_info.get("gpuCount")
            values["nvLinkDomainUid"] = gpu_info.get("nvLinkDomainUid")
            values["nvLinkCliqueId"] = gpu_info.get("nvLinkCliqueId")
        return values

    def __repr__(self) -> str:
        """Return a string representation of the RunAINode object."""
        return (
            f"RunAINode(name={self.name!r}, id={self.id!r}, cluster_uuid={self.cluster_uuid!r}, "
            f"node_pool={self.node_pool!r}, gpu_type={self.gpu_type!r}, gpu_count={self.gpu_count}, "
            f"nvlink_domain_uid={self.nvlink_domain_uid!r}, nvlink_clique_id={self.nvlink_clique_id!r}, "
            f"status={self.status.value!r})"
        )
