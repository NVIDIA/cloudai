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
from typing import Any, Dict


class InvolvedObject:
    """Represent an object involved in a RunAI event."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.uid: str = data.get("uid", "")
        self.kind: str = data.get("kind", "")
        self.name: str = data.get("name", "")
        self.namespace: str = data.get("namespace", "")

    def __repr__(self) -> str:
        """Return a string representation of the InvolvedObject."""
        return f"InvolvedObject(uid={self.uid}, kind={self.kind}, name={self.name}, namespace={self.namespace})"


class RunAIEvent:
    """Represent a RunAI event."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.created_at: datetime = datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))
        self.id: str = data.get("id", "")
        self.type: str = data.get("type", "")
        self.cluster_id: str = data.get("clusterId", "")
        self.message: str = data.get("message", "")
        self.reason: str = data.get("reason", "")
        self.source: str = data.get("source", "")
        self.involved_object: InvolvedObject = InvolvedObject(data.get("involvedObject", {}))

    def __repr__(self) -> str:
        """Return a string representation of the RunAIEvent."""
        return (
            f"RunAIEvent(created_at={self.created_at}, id={self.id}, type={self.type}, "
            f"cluster_id={self.cluster_id}, message={self.message}, reason={self.reason}, "
            f"source={self.source}, involved_object={self.involved_object})"
        )
