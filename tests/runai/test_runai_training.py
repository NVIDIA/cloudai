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

import pytest

from cloudai.systems.runai.runai_training import ActualPhase, RunAITraining, TrainingSpec, WorkloadPhase


@pytest.fixture
def training_data() -> Dict[str, Any]:
    return {
        "name": "test-training",
        "requestedName": "test-training-request",
        "workloadId": "workload-123",
        "projectId": "project-456",
        "clusterId": "cluster-789",
        "createdBy": "user@example.com",
        "createdAt": "2025-01-01T00:00:00Z",
        "deletedAt": None,
        "desiredPhase": "Running",
        "actualPhase": "Initializing",
        "spec": {
            "command": "echo Hello World",
            "args": "--verbose",
            "image": "test-image",
            "restartPolicy": "Always",
            "nodePools": ["pool1", "pool2"],
            "compute": {
                "gpuDevicesRequest": 1,
                "cpuCoreRequest": 2.0,
            },
            "storage": {"pvc": [{"claimName": "pvc1", "path": "/mnt/data"}]},
            "security": {"runAsUid": 1000, "allowPrivilegeEscalation": False},
            "completions": 1,
            "parallelism": 1,
        },
    }


def test_training_initialization(training_data: Dict[str, Any]) -> None:
    training = RunAITraining(**training_data)
    assert training.name == training_data["name"]
    assert training.requested_name == training_data["requestedName"]
    assert training.workload_id == training_data["workloadId"]
    assert training.project_id == training_data["projectId"]
    assert training.cluster_id == training_data["clusterId"]
    assert training.created_by == training_data["createdBy"]
    assert training.created_at == datetime.fromisoformat(training_data["createdAt"].replace("Z", "+00:00"))
    assert training.deleted_at is None
    assert training.desired_phase == WorkloadPhase.RUNNING
    assert training.actual_phase == ActualPhase.INITIALIZING
    assert isinstance(training.spec, TrainingSpec)


@pytest.mark.parametrize(
    "desired_phase, actual_phase",
    [
        ("Running", "Initializing"),
        ("Stopped", "Failed"),
        ("Deleted", "Terminating"),
    ],
)
def test_training_phases(training_data: Dict[str, Any], desired_phase: str, actual_phase: str) -> None:
    training_data["desiredPhase"] = desired_phase
    training_data["actualPhase"] = actual_phase
    training = RunAITraining(**training_data)
    assert training.desired_phase == WorkloadPhase(desired_phase)
    assert training.actual_phase == ActualPhase(actual_phase)


def test_training_specification(training_data: Dict[str, Any]) -> None:
    training = RunAITraining(**training_data)
    spec = training.spec
    assert spec is not None
    assert spec.command == training_data["spec"]["command"]
    assert spec.args == training_data["spec"]["args"]
    assert spec.image == training_data["spec"]["image"]
    assert spec.restart_policy == training_data["spec"]["restartPolicy"]
    assert spec.node_pools == training_data["spec"]["nodePools"]
    assert spec.compute is not None
    assert spec.compute.gpu_devices_request == training_data["spec"]["compute"]["gpuDevicesRequest"]
    assert spec.compute.cpu_core_request == training_data["spec"]["compute"]["cpuCoreRequest"]
    assert spec.storage is not None
    assert spec.storage.pvc[0].claim_name == training_data["spec"]["storage"]["pvc"][0]["claimName"]
    assert spec.security is not None
    assert spec.security.run_as_uid == training_data["spec"]["security"]["runAsUid"]
    assert spec.security.allow_privilege_escalation == training_data["spec"]["security"]["allowPrivilegeEscalation"]
