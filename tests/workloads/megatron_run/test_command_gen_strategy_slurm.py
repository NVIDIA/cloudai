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

from cloudai import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.megatron_run import (
    MegatronRunCmdArgs,
    MegatronRunSlurmCommandGenStrategy,
    MegatronRunTestDefinition,
)


def _megatron_run_test_run(tmp_path: Path, command_prefix: str = "") -> TestRun:
    tdef = MegatronRunTestDefinition(
        name="megatron-run",
        description="MegatronRun",
        test_template_name="MegatronRun",
        cmd_args=MegatronRunCmdArgs(
            docker_image_url="fake://url/mr",
            run_script=tmp_path / "pretrain.py",
            command_prefix=command_prefix,
        ),
    )
    return TestRun(name="megatron-run", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)


def test_megatron_run_command_has_no_prefix_by_default(slurm_system: SlurmSystem, tmp_path: Path) -> None:
    tr = _megatron_run_test_run(tmp_path)
    cmd_gen = MegatronRunSlurmCommandGenStrategy(slurm_system, tr)

    command = cmd_gen.generate_test_command()

    assert command[0] == "python"
    assert "--command-prefix" not in " ".join(command)


def test_megatron_run_command_prefix_wraps_python_command(slurm_system: SlurmSystem, tmp_path: Path) -> None:
    tr = _megatron_run_test_run(tmp_path, command_prefix="/cloudai_install/bindpcie --")
    cmd_gen = MegatronRunSlurmCommandGenStrategy(slurm_system, tr)

    command = cmd_gen.generate_test_command()
    srun_command = cmd_gen.gen_srun_command()

    assert command[:2] == ["/cloudai_install/bindpcie --", "python"]
    assert "--command-prefix" not in " ".join(command)
    assert "/cloudai_install/bindpcie -- python " in srun_command
