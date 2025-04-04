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

from pydantic import BaseModel, Field


class InvolvedObject(BaseModel):
    """Represent an object involved in a RunAI event."""

    uid: str = Field(default="", alias="uid")
    kind: str = Field(default="", alias="kind")
    name: str = Field(default="", alias="name")
    namespace: str = Field(default="", alias="namespace")


class RunAIEvent(BaseModel):
    """Represent a RunAI event."""

    created_at: datetime = Field(alias="createdAt")
    id: str = Field(default="", alias="id")
    type: str = Field(default="", alias="type")
    cluster_id: str = Field(default="", alias="clusterId")
    message: str = Field(default="", alias="message")
    reason: str = Field(default="", alias="reason")
    source: str = Field(default="", alias="source")
    involved_object: InvolvedObject = Field(default_factory=InvolvedObject, alias="involvedObject")

    class Config:
        """Pydantic configuration for RunAIEvent."""

        populate_by_name = True
        populate_by_alias = True
        validate_by_alias = True

    def __repr__(self) -> str:
        """Return a string representation of the RunAIEvent."""
        return (
            f"RunAIEvent(created_at={self.created_at}, id={self.id}, type={self.type}, "
            f"cluster_id={self.cluster_id}, message={self.message}, reason={self.reason}, "
            f"source={self.source}, involved_object={self.involved_object})"
        )
