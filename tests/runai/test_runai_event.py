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

from datetime import datetime, timezone
from typing import Any, Dict

import pytest

from cloudai.systems.runai.runai_event import InvolvedObject, RunAIEvent


@pytest.fixture
def involved_object_data() -> Dict[str, Any]:
    return {"uid": "abc-123", "kind": "Pod", "name": "example-pod", "namespace": "default"}


@pytest.fixture
def runai_event_data(involved_object_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "createdAt": "2025-04-01T12:00:00Z",
        "id": "event-001",
        "type": "Warning",
        "clusterId": "cluster-123",
        "message": "GPU resource exhausted",
        "reason": "OutOfGpu",
        "source": "scheduler",
        "involvedObject": involved_object_data,
    }


def test_involved_object_initialization(involved_object_data: Dict[str, Any]) -> None:
    obj = InvolvedObject(involved_object_data)
    assert obj.uid == involved_object_data["uid"]
    assert obj.kind == involved_object_data["kind"]
    assert obj.name == involved_object_data["name"]
    assert obj.namespace == involved_object_data["namespace"]


def test_involved_object_defaults() -> None:
    obj = InvolvedObject({})
    assert obj.uid == ""
    assert obj.kind == ""
    assert obj.name == ""
    assert obj.namespace == ""


def test_runai_event_initialization(runai_event_data: Dict[str, Any]) -> None:
    event = RunAIEvent(runai_event_data)
    expected_datetime = datetime(2025, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert event.created_at == expected_datetime
    assert event.id == runai_event_data["id"]
    assert event.type == runai_event_data["type"]
    assert event.cluster_id == runai_event_data["clusterId"]
    assert event.message == runai_event_data["message"]
    assert event.reason == runai_event_data["reason"]
    assert event.source == runai_event_data["source"]
    assert isinstance(event.involved_object, InvolvedObject)
    assert event.involved_object.uid == runai_event_data["involvedObject"]["uid"]


@pytest.mark.parametrize(
    "field,expected", [("id", ""), ("type", ""), ("cluster_id", ""), ("message", ""), ("reason", ""), ("source", "")]
)
def test_runai_event_defaults(field: str, expected: str) -> None:
    minimal_data = {"createdAt": "2025-04-01T12:00:00Z"}
    event = RunAIEvent(minimal_data)
    assert getattr(event, field) == expected
    assert isinstance(event.involved_object, InvolvedObject)
