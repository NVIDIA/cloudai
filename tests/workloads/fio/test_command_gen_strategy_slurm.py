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
                "max-jobs": 4,
                "--warnings-fatal": True,
                "group_reporting": True,
            },
            docker_image_url="openeuler/fio:3.42-oe2403sp3",
            num_tasks_per_node=2,
        ),
    )
    return TestRun(name="fio", test=tdef, num_nodes=2, nodes=[], output_path=tmp_path / "output")


def test_slurm_typical_cluster_execution_command(slurm_system: SlurmSystem, fio_tr: TestRun) -> None:
    fio_tr.output_path.mkdir(parents=True, exist_ok=True)
    strategy = FioSlurmCommandGenStrategy(slurm_system, fio_tr)

    assert strategy.generate_test_command() == [
        "/tmp/fio/fio",
        "--name=randwrite",
        "--filename=/dev/ng4n1",
        "--rw=randwrite",
        "--bs=128k",
        "--iodepth=8",
        "--max-jobs=4",
        "--warnings-fatal",
        "--group_reporting",
    ]
    assert not fio_tr.test.is_dse_job

    command = strategy.gen_srun_command()
    assert "--mpi=" not in command
    assert "-N2" in command
    assert "--ntasks-per-node=2" in command
    assert "--ntasks=4" in command
    assert "--container-image=openeuler/fio:3.42-oe2403sp3" in command
    assert "/tmp/fio/fio --name=randwrite" in command
    assert "IOPS=" in strategy.gen_srun_success_check()


def test_job_file_is_appended_without_declaring_fio_options(slurm_system: SlurmSystem, tmp_path: Path) -> None:
    tdef = FioTestDefinition(
        name="fio",
        description="fio test",
        test_template_name="Fio",
        cmd_args=FioCmdArgs(fio_binary="fio", job_file="/tmp/kv_emulation.fio"),
    )
    tr = TestRun(name="fio", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")
    strategy = FioSlurmCommandGenStrategy(slurm_system, tr)

    assert strategy.generate_test_command() == ["fio", "/tmp/kv_emulation.fio"]


def test_nested_arg_table_repeats_same_fio_option(slurm_system: SlurmSystem, tmp_path: Path) -> None:
    tdef = FioTestDefinition(
        name="fio",
        description="fio test",
        test_template_name="Fio",
        cmd_args=FioCmdArgs(
            args={
                "name": "repeat",
                "a": {"0": "=foo", "1": "bar"},
                "--client": {"0": "host1", "1": "host2"},
            },
        ),
    )
    tr = TestRun(name="fio", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")
    strategy = FioSlurmCommandGenStrategy(slurm_system, tr)

    assert strategy.generate_test_command() == [
        "fio",
        "--name=repeat",
        "--a==foo",
        "--a=bar",
        "--client=host1",
        "--client=host2",
    ]
