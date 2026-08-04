# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import shlex
from typing import cast

import pytest

from cloudai.core import TestRun, TestScenario
from cloudai.models.workload import NsysConfiguration
from cloudai.systems.slurm import SingleSbatchRunner, SlurmSystem
from cloudai.workloads.slurm_container import (
    SlurmContainerCmdArgs,
    SlurmContainerCommandGenStrategy,
    SlurmContainerTestDefinition,
)
from cloudai.workloads.slurm_container.slurm_container import EXIT_CODE_FILE_NAME


def _status_capture(test_run: TestRun) -> str:
    exit_code_path = shlex.quote(str((test_run.output_path / EXIT_CODE_FILE_NAME).absolute()))
    return (
        "; rc=$?; "
        f"""printf '%s\\n' "$rc" > {exit_code_path}; """
        """(exit "$rc")"""
    )


@pytest.fixture
def test_run() -> TestRun:
    tdef = SlurmContainerTestDefinition(
        name="sc",
        description="desc",
        test_template_name="tt",
        cmd_args=SlurmContainerCmdArgs(docker_image_url="docker://url", cmd="cmd"),
    )
    tr = TestRun(name="name", test=tdef, num_nodes=1, nodes=[])
    return tr


def test_default(slurm_system: SlurmSystem, test_run: TestRun) -> None:
    cgs = SlurmContainerCommandGenStrategy(slurm_system, test_run)
    cmd = cgs.gen_srun_command()
    srun_part = (
        f"srun --export=ALL --mpi={slurm_system.mpi} -N{test_run.num_nodes} "
        f"--container-image={test_run.test.cmd_args.docker_image_url} "
        f"--container-mounts={test_run.output_path.absolute()}:/cloudai_run_results,"
        f"{slurm_system.install_path.absolute()}:/cloudai_install,"
        f"{test_run.output_path.absolute()} "
        f"--no-container-mount-home"
    )

    assert cmd == (
        f'{srun_part} bash -c "source {(test_run.output_path / "env_vars.sh").absolute()}; cmd"'
        f"{_status_capture(test_run)}"
    )


def test_with_nsys(slurm_system: SlurmSystem, test_run: TestRun) -> None:
    cgs = SlurmContainerCommandGenStrategy(slurm_system, test_run)
    nsys = NsysConfiguration()
    test_run.test.nsys = nsys
    cmd = cgs.gen_srun_command()

    srun_part = (
        f"srun --export=ALL --mpi={slurm_system.mpi} -N{test_run.num_nodes} "
        f"--container-image={test_run.test.cmd_args.docker_image_url} "
        f"--container-mounts={test_run.output_path.absolute()}:/cloudai_run_results,"
        f"{slurm_system.install_path.absolute()}:/cloudai_install,"
        f"{test_run.output_path.absolute()} "
        f"--no-container-mount-home"
    )

    assert cmd == (
        f'{srun_part} bash -c "source {(test_run.output_path / "env_vars.sh").absolute()}; nsys profile cmd"'
        f"{_status_capture(test_run)}"
    )


def test_with_extra_srun_args(slurm_system: SlurmSystem, test_run: TestRun) -> None:
    extra_args = ["--ntasks=1", "--ntasks-per-node=1"]
    tdef = cast(SlurmContainerTestDefinition, test_run.test)
    tdef.extra_srun_args = extra_args

    cgs = SlurmContainerCommandGenStrategy(slurm_system, test_run)
    cmd = cgs.gen_srun_command()

    srun_part = (
        f"srun --export=ALL --mpi={slurm_system.mpi} -N{test_run.num_nodes} "
        f"--container-image={test_run.test.cmd_args.docker_image_url} "
        f"--container-mounts={test_run.output_path.absolute()}:/cloudai_run_results,"
        f"{slurm_system.install_path.absolute()}:/cloudai_install,"
        f"{test_run.output_path.absolute()} "
        f"--no-container-mount-home "
        f"{' '.join(extra_args)}"
    )

    assert cmd == (
        f'{srun_part} bash -c "source {(test_run.output_path / "env_vars.sh").absolute()}; cmd"'
        f"{_status_capture(test_run)}"
    )


def test_single_sbatch_writes_exit_code_to_per_test_output(slurm_system: SlurmSystem, test_run: TestRun) -> None:
    test_run.output_path = slurm_system.output_path / "single-batch"
    test_run.output_path.mkdir(parents=True)
    scenario = TestScenario(name="tc", test_runs=[test_run])
    runner = SingleSbatchRunner(
        mode="run",
        system=slurm_system,
        test_scenario=scenario,
        output_path=slurm_system.output_path,
    )

    block = runner.get_single_tr_block(test_run)

    assert f"{test_run.output_path.absolute()}:/cloudai_run_results" in block
    assert str((test_run.output_path / EXIT_CODE_FILE_NAME).absolute()) in block
    assert f"/cloudai_run_results/{EXIT_CODE_FILE_NAME}" not in block


def test_multi_task_run_records_one_aggregate_srun_status(slurm_system: SlurmSystem, test_run: TestRun) -> None:
    tdef = cast(SlurmContainerTestDefinition, test_run.test)
    tdef.extra_srun_args = ["--ntasks=2"]
    tdef.cmd_args.cmd = r"bash -c 'exit \$SLURM_PROCID'"
    cgs = SlurmContainerCommandGenStrategy(slurm_system, test_run)

    cmd = cgs.gen_srun_command()
    exit_code_path = str((test_run.output_path / EXIT_CODE_FILE_NAME).absolute())

    assert "--ntasks=2" in cmd
    assert r"""bash -c 'exit \$SLURM_PROCID'"; rc=$?;""" in cmd
    assert cmd.count(exit_code_path) == 1
