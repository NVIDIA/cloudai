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
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy


@pytest.fixture
def strategy_fixture(slurm_system: SlurmSystem) -> SlurmCommandGenStrategy:
    cmd_args = {"test_arg": "test_value"}
    strategy = SlurmCommandGenStrategy(slurm_system, cmd_args)
    return strategy


def test_filename_generation(strategy_fixture: SlurmCommandGenStrategy, tmp_path: Path):
    args = {"job_name": "test_job", "num_nodes": 2, "partition": "test_partition", "node_list_str": "node1,node2"}
    env_vars = {"TEST_VAR": "VALUE"}
    srun_command = "srun --test test_arg"
    output_path = tmp_path

    sbatch_command = strategy_fixture._write_sbatch_script(args, env_vars, srun_command, output_path)
    filepath_from_command = sbatch_command.split()[-1]

    assert output_path.joinpath("cloudai_sbatch_script.sh").exists()

    with open(filepath_from_command, "r") as file:
        file_contents = file.read()
    assert "test_job" in file_contents
    assert "node1,node2" in file_contents
    assert "srun --test test_arg" in file_contents

    assert sbatch_command == f"sbatch {filepath_from_command}"


def test_num_nodes_and_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    nodes = ["node1", "node2"]
    num_nodes = 3

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)

    assert slurm_args["num_nodes"] == len(nodes)


def test_only_num_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    nodes = []
    num_nodes = 3

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)

    assert slurm_args["num_nodes"] == num_nodes


def test_only_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    nodes = ["node1", "node2"]
    num_nodes = 0

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)

    assert slurm_args["num_nodes"] == len(nodes)


def test_raises_if_no_default_partition(slurm_system: SlurmSystem):
    slurm_system.default_partition = ""
    with pytest.raises(ValueError) as exc_info:
        SlurmCommandGenStrategy(slurm_system, {})
    assert (
        "Default partition not set in the Slurm system object. "
        "The 'default_partition' attribute should be properly defined in the Slurm "
        "system configuration. Please ensure that 'default_partition' is set correctly "
        "in the corresponding system configuration (e.g., system.toml)."
    ) in str(exc_info.value)
