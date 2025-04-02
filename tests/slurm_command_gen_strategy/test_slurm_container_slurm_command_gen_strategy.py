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

from pathlib import Path

import pytest

from cloudai import TestRun
from cloudai._core.test import Test
from cloudai._core.test_template import TestTemplate
from cloudai.models.workload import NsysConfiguration
from cloudai.systems import SlurmSystem
from cloudai.workloads.slurm_container import (
    SlurmContainerCmdArgs,
    SlurmContainerCommandGenStrategy,
    SlurmContainerTestDefinition,
)


@pytest.fixture
def test_run(slurm_system: SlurmSystem) -> TestRun:
    tdef = SlurmContainerTestDefinition(
        name="sc",
        description="desc",
        test_template_name="tt",
        cmd_args=SlurmContainerCmdArgs(docker_image_url="docker://url", cmd="cmd"),
    )
    t = Test(test_definition=tdef, test_template=TestTemplate(system=slurm_system))
    tr = TestRun(name="name", test=t, num_nodes=1, nodes=[])
    return tr


def test_default(slurm_system: SlurmSystem, test_run: TestRun) -> None:
    cgs = SlurmContainerCommandGenStrategy(slurm_system, {})
    cmd = cgs.gen_srun_command(test_run)
    srun_part = (
        f"srun --export=ALL --mpi={slurm_system.mpi} "
        f"--container-image={test_run.test.test_definition.cmd_args.docker_image_url} "
        f"--container-mounts={Path.cwd().absolute()}:/cloudai_run_results,"
        f"{slurm_system.install_path.absolute()}:/cloudai_install "
        f"--no-container-mount-home"
    )

    assert cmd == f'{srun_part} bash -c "cmd"'


def test_with_nsys(slurm_system: SlurmSystem, test_run: TestRun) -> None:
    cgs = SlurmContainerCommandGenStrategy(slurm_system, {})
    nsys = NsysConfiguration()
    test_run.test.test_definition.nsys = nsys
    cmd = cgs.gen_srun_command(test_run)

    srun_part = (
        f"srun --export=ALL --mpi={slurm_system.mpi} "
        f"--container-image={test_run.test.test_definition.cmd_args.docker_image_url} "
        f"--container-mounts={Path.cwd().absolute()}:/cloudai_run_results,"
        f"{slurm_system.install_path.absolute()}:/cloudai_install "
        f"--no-container-mount-home"
    )

    assert cmd == f'{srun_part} bash -c "{" ".join(nsys.cmd_args)} cmd"'
