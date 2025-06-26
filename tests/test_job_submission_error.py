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
from unittest.mock import Mock

import pytest

from cloudai.core import JobIdRetrievalError, Test, TestRun, TestScenario, TestTemplate
from cloudai.systems.slurm import SlurmRunner, SlurmSystem
from cloudai.util import CommandShell
from cloudai.workloads.sleep.sleep import SleepCmdArgs, SleepTestDefinition
from cloudai.workloads.sleep.slurm_command_gen_strategy import SleepSlurmCommandGenStrategy


class MockCommandShell(CommandShell):
    def execute(self, command):
        mock_popen = Mock(spec=subprocess.Popen)
        mock_popen.communicate.return_value = (
            "",
            "sbatch: error: Batch job submission failed: Requested node configuration is not available",
        )
        return mock_popen


@pytest.fixture
def slurm_runner(slurm_system: SlurmSystem, tmp_path: Path) -> SlurmRunner:
    test_scenario = TestScenario(
        name="Test Scenario",
        test_runs=[
            TestRun(
                "tr-name",
                Test(
                    test_definition=SleepTestDefinition(
                        name="n", description="d", test_template_name="Sleep", cmd_args=SleepCmdArgs()
                    ),
                    test_template=TestTemplate(slurm_system),
                ),
                1,
                [],
                output_path=slurm_system.output_path / "tr-name",
            )
        ],
    )
    test_scenario.test_runs[0].output_path.mkdir(parents=True, exist_ok=True)
    test_scenario.test_runs[0].test.test_template.command_gen_strategy = SleepSlurmCommandGenStrategy(slurm_system, {})
    runner = SlurmRunner(mode="run", system=slurm_system, test_scenario=test_scenario, output_path=tmp_path)
    runner.cmd_shell = MockCommandShell()
    return runner


def test_job_id_retrieval_error(slurm_runner: SlurmRunner):
    tr = slurm_runner.test_scenario.test_runs[0]
    with pytest.raises(JobIdRetrievalError) as excinfo:
        slurm_runner._submit_test(tr)
    assert "Failed to retrieve job ID." in str(excinfo.value)
    assert "sbatch: error: Batch job submission failed: Requested node configuration is not available" in str(
        excinfo.value
    )
