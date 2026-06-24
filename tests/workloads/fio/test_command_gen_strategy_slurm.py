# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.fio import FioCmdArgs, FioSlurmCommandGenStrategy, FioTestDefinition


@pytest.fixture
def fio_tr(slurm_system: SlurmSystem, tmp_path: Path) -> TestRun:
    tdef = FioTestDefinition(
        name="fio",
        description="fio test",
        test_template_name="Fio",
        cmd_args=FioCmdArgs(
            fio_binary="/tmp/fio/fio",
            args={
                "name": "randwrite",
                "filename": "/dev/ng4n1",
                "rw": "randwrite",
                "bs": "128k",
                "iodepth": 8,
                "group_reporting": True,
            },
            num_tasks_per_node=2,
        ),
    )
    return TestRun(name="fio", test=tdef, num_nodes=2, nodes=[], output_path=tmp_path / "output")


def test_generate_test_command_cli_args(slurm_system: SlurmSystem, fio_tr: TestRun) -> None:
    strategy = FioSlurmCommandGenStrategy(slurm_system, fio_tr)

    assert strategy.generate_test_command() == [
        "/tmp/fio/fio",
        "--name=randwrite",
        "--filename=/dev/ng4n1",
        "--rw=randwrite",
        "--bs=128k",
        "--iodepth=8",
        "--group-reporting",
    ]


def test_generate_test_command_job_file(slurm_system: SlurmSystem, tmp_path: Path) -> None:
    tdef = FioTestDefinition(
        name="fio",
        description="fio test",
        test_template_name="Fio",
        cmd_args=FioCmdArgs(fio_binary="fio", job_file="/tmp/kv_emulation.fio"),
    )
    tr = TestRun(name="fio", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")
    strategy = FioSlurmCommandGenStrategy(slurm_system, tr)

    assert strategy.generate_test_command() == ["fio", "/tmp/kv_emulation.fio"]


def test_gen_srun_prefix_omits_mpi(slurm_system: SlurmSystem, fio_tr: TestRun) -> None:
    strategy = FioSlurmCommandGenStrategy(slurm_system, fio_tr)

    prefix = strategy.gen_srun_prefix()

    assert "srun" in prefix
    assert not any(part.startswith("--mpi=") for part in prefix)


def test_gen_srun_command_adds_multinode_task_counts(slurm_system: SlurmSystem, fio_tr: TestRun) -> None:
    fio_tr.output_path.mkdir(parents=True, exist_ok=True)
    strategy = FioSlurmCommandGenStrategy(slurm_system, fio_tr)

    command = strategy.gen_srun_command()

    assert "--mpi=" not in command
    assert "--ntasks-per-node=2" in command
    assert "--ntasks=4" in command
    assert "/tmp/fio/fio --name=randwrite" in command


def test_gen_srun_success_check_uses_fio_summary_markers(slurm_system: SlurmSystem, fio_tr: TestRun) -> None:
    strategy = FioSlurmCommandGenStrategy(slurm_system, fio_tr)

    assert "IOPS=" in strategy.gen_srun_success_check()
    assert "BW=" in strategy.gen_srun_success_check()


def test_default_passthrough_args_do_not_create_dse_job(fio_tr: TestRun) -> None:
    assert not fio_tr.test.is_dse_job


def test_list_valued_fio_args_create_dse_job(tmp_path: Path) -> None:
    tdef = FioTestDefinition(
        name="fio",
        description="fio test",
        test_template_name="Fio",
        cmd_args=FioCmdArgs(args={"name": "sweep", "iodepth": [1, 8]}),
    )
    tr = TestRun(name="fio", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")

    assert tr.is_dse_job
    assert tr.param_space == {"args.iodepth": [1, 8]}
