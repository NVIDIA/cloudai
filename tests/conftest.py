# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path

import pytest
from cloudai.systems.slurm.slurm_system import SlurmPartition, SlurmSystem


@pytest.fixture
def slurm_system() -> SlurmSystem:
    system = SlurmSystem(
        name="test_system",
        install_path=Path("/fake/path"),
        output_path=Path("/fake/output"),
        default_partition="main",
        partitions=[
            SlurmPartition(name="main", nodes=["node-[033-064]"]),
            SlurmPartition(name="backup", nodes=["node0[1-8]"]),
        ],
    )
    return system
