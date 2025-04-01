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

from cloudai.systems.runai.runai_pvc import RunAIPVC


@pytest.fixture
def raw_pvc() -> Dict[str, Any]:
    return {
        "meta": {
            "id": "pvc-123",
            "name": "test-pvc",
            "kind": "PersistentVolumeClaim",
            "scope": "namespace",
            "clusterId": "cluster-001",
            "tenantId": 42,
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-02T00:00:00Z",
            "deletedAt": None,
            "createdBy": "user-1",
            "updatedBy": "user-2",
            "description": "test pvc",
            "autoDelete": True,
            "projectId": 1001,
        },
        "spec": {
            "claimName": "claim-001",
            "path": "/mnt/data",
            "readOnly": False,
            "ephemeral": True,
            "existingPvc": True,
            "claimInfo": {
                "accessModes": {"ReadWriteOnce": True},
                "size": "10Gi",
                "storageClass": "standard",
                "volumeMode": "Filesystem",
            },
        },
        "clusterInfo": {"resources": [{"cpu": "4", "memory": "16Gi"}]},
    }


def test_pvc_initialization(raw_pvc: Dict[str, Any]) -> None:
    pvc = RunAIPVC(raw_pvc)
    assert pvc.name == "test-pvc"
    assert pvc.description == "test pvc"
    assert pvc.scope == "namespace"
    assert pvc.cluster_id == "cluster-001"
    assert pvc.department_id is None
    assert pvc.project_id == 1001
    assert pvc.auto_delete is True
    assert pvc.workload_supported_types is None
    assert pvc.id == "pvc-123"
    assert pvc.kind == "PersistentVolumeClaim"
    assert pvc.tenant_id == 42
    assert pvc.created_by == "user-1"
    assert pvc.created_at == "2024-01-01T00:00:00Z"
    assert pvc.updated_by == "user-2"
    assert pvc.updated_at == "2024-01-02T00:00:00Z"
    assert pvc.deleted_at is None
    assert pvc.deleted_by is None
    assert pvc.project_name is None
    assert pvc.update_count is None
    assert pvc.claim_name == "claim-001"
    assert pvc.path == "/mnt/data"
    assert pvc.read_only is False
    assert pvc.ephemeral is True
    assert pvc.existing_pvc is True
    assert pvc.access_modes == {"ReadWriteOnce": True}
    assert pvc.size == "10Gi"
    assert pvc.storage_class == "standard"
    assert pvc.volume_mode == "Filesystem"
    assert pvc.resources == [{"cpu": "4", "memory": "16Gi"}]


@pytest.mark.parametrize(
    "optional_fields",
    [
        {},
        {"ephemeral": False, "existingPvc": False},
        {"claimInfo": {"accessModes": None, "size": None, "storageClass": None, "volumeMode": None}},
    ],
)
def test_pvc_optional_fields(raw_pvc: Dict[str, Any], optional_fields: Dict[str, Any]) -> None:
    raw_pvc_mod = raw_pvc.copy()
    raw_pvc_mod["spec"] = {**raw_pvc["spec"], **optional_fields}
    pvc = RunAIPVC(raw_pvc_mod)
    assert isinstance(pvc, RunAIPVC)


def test_pvc_repr(raw_pvc: Dict[str, Any]) -> None:
    pvc = RunAIPVC(raw_pvc)
    expected = (
        "RunAIPVC(name='test-pvc', id='pvc-123', created_by='user-1', "
        "scope='namespace', cluster_id='cluster-001', project_id=1001, "
        "size='10Gi', storage_class='standard', path='/mnt/data', "
        "claim_name='claim-001')"
    )
    assert repr(pvc) == expected
