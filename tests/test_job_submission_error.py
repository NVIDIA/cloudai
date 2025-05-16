# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from cloudai import Test, TestRun, TestScenario, TestTemplate
from cloudai.exceptions import JobIdRetrievalError
from cloudai.systems.slurm import SlurmPartition, SlurmRunner, SlurmSystem
from cloudai.util import CommandShell


class MockCommandShell(CommandShell):
    def execute(self, command):
        mock_popen = Mock(spec=subprocess.Popen)
        mock_popen.communicate.return_value = (
            "",
            "sbatch: error: Batch job submission failed: Requested node configuration is not available",
        )
        return mock_popen


class MockTest(Test):
    def __init__(self, section_name):
        self.test_template = MagicMock(spec=TestTemplate)
        self.test_template.get_job_id.return_value = None
        self.env_vars = {}
        self.section_name = "Tests.1"
        self.current_iteration = 0


@pytest.fixture
def slurm_system(tmp_path: Path):
    system = SlurmSystem(
        name="test_system",
        install_path=tmp_path,
        output_path=tmp_path,
        default_partition="main",
        partitions=[SlurmPartition(name="main")],
    )
    return system


@pytest.fixture
def slurm_runner(slurm_system, tmp_path: Path) -> SlurmRunner:
    test_scenario = TestScenario(
        name="Test Scenario", test_runs=[TestRun("tr-name", MockTest(section_name="Mock Test"), 1, [])]
    )
    runner = SlurmRunner(mode="run", system=slurm_system, test_scenario=test_scenario, output_path=tmp_path)
    runner.cmd_shell = MockCommandShell()
    return runner


def test_job_id_retrieval_error(slurm_runner: SlurmRunner):
    tr = slurm_runner.test_scenario.test_runs[0]
    with pytest.raises(JobIdRetrievalError) as excinfo:
        slurm_runner._submit_test(tr)
    assert "Failed to retrieve job ID from command output." in str(excinfo.value)
    assert "sbatch: error: Batch job submission failed: Requested node configuration is not available" in str(
        excinfo.value
    )
