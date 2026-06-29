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

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.fio import FioCmdArgs, FioSlurmCommandGenStrategy, FioTestDefinition


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


def test_success_check_matches_fio_summary_lines(slurm_system: SlurmSystem, tmp_path: Path) -> None:
    tdef = FioTestDefinition(
        name="fio",
        description="fio test",
        test_template_name="Fio",
        cmd_args=FioCmdArgs(),
    )
    tr = TestRun(name="fio", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")
    strategy = FioSlurmCommandGenStrategy(slurm_system, tr)

    assert "IOPS=.*BW=|BW=.*IOPS=" in strategy.gen_srun_success_check()


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
