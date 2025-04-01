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

from typing import Any, Dict

import pytest

from cloudai.systems.runai.runai_project import RunAIProject


@pytest.fixture
def example_data() -> Dict[str, Any]:
    return {
        "description": "string",
        "schedulingRules": {
            "interactiveJobTimeLimitSeconds": 100,
            "interactiveJobMaxIdleDurationSeconds": 100,
            "interactiveJobPreemptIdleDurationSeconds": 100,
            "trainingJobMaxIdleDurationSeconds": 100,
            "trainingJobTimeLimitSeconds": 100,
        },
        "defaultNodePools": ["string"],
        "nodeTypes": {
            "training": ["string"],
            "workspace": ["string"],
            "names": {"property1": "string", "property2": "string"},
        },
        "resources": [
            {
                "nodePool": {"id": 22, "name": "default"},
                "gpu": {"deserved": 1000, "limit": 0, "overQuotaWeight": 2},
                "cpu": {"deserved": 1000, "limit": 0, "overQuotaWeight": 2},
                "memory": {"deserved": 1000, "limit": 0, "overQuotaWeight": 2, "units": "Mib"},
                "priority": "Normal",
            }
        ],
        "name": "organization1",
        "clusterId": "71f69d83-ba66-4822-adf5-55ce55efd210",
        "requestedNamespace": "runai-proj1",
        "enforceRunaiScheduler": True,
        "parentId": "53a9228e-a722-420d-a102-9dc90da2efca",
    }


def test_project_initialization(example_data: Dict[str, Any]) -> None:
    project = RunAIProject(example_data)
    assert project.name == "organization1"
    assert project.cluster_id == "71f69d83-ba66-4822-adf5-55ce55efd210"
    assert project.description == "string"
    assert project.scheduling_rules == {
        "interactiveJobTimeLimitSeconds": 100,
        "interactiveJobMaxIdleDurationSeconds": 100,
        "interactiveJobPreemptIdleDurationSeconds": 100,
        "trainingJobMaxIdleDurationSeconds": 100,
        "trainingJobTimeLimitSeconds": 100,
    }
    assert project.default_node_pools == ["string"]
    assert project.node_types == {
        "training": ["string"],
        "workspace": ["string"],
        "names": {"property1": "string", "property2": "string"},
    }
    assert project.resources == [
        {
            "nodePool": {"id": 22, "name": "default"},
            "gpu": {"deserved": 1000, "limit": 0, "overQuotaWeight": 2},
            "cpu": {"deserved": 1000, "limit": 0, "overQuotaWeight": 2},
            "memory": {"deserved": 1000, "limit": 0, "overQuotaWeight": 2, "units": "Mib"},
            "priority": "Normal",
        }
    ]
    assert project.requested_namespace == "runai-proj1"
    assert project.enforce_runai_scheduler is True
    assert project.parent_id == "53a9228e-a722-420d-a102-9dc90da2efca"


def test_repr_output(example_data: Dict[str, Any]) -> None:
    project = RunAIProject(example_data)
    output = repr(project)
    assert "RunAIProject" in output
    assert "organization1" in output
    assert "71f69d83-ba66-4822-adf5-55ce55efd210" in output
