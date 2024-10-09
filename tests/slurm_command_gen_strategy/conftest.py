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
from cloudai.systems import SlurmSystem
from cloudai.systems.slurm import SlurmNodeState
from cloudai.systems.slurm.slurm_system import SlurmPartition


@pytest.fixture
def slurm_system(tmp_path: Path) -> SlurmSystem:
    slurm_system = SlurmSystem(
        name="TestSystem",
        install_path=tmp_path / "install",
        output_path=tmp_path / "output",
        default_partition="main",
        partitions=[
            SlurmPartition(name="main", nodes=["node[1-4]"]),
        ],
        mpi="fake-mpi",
    )
    for node in slurm_system.partitions[0].slurm_nodes:
        node.state = SlurmNodeState.IDLE
    Path(slurm_system.install_path).mkdir()
    Path(slurm_system.output_path).mkdir()
    return slurm_system
