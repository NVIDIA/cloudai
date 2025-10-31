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

import argparse

import pytest

from cloudai.cli.handlers import handle_dse_job
from cloudai.core import Runner, TestDependency, TestRun, TestScenario
from cloudai.systems.slurm.slurm_system import SlurmSystem


@pytest.mark.parametrize("dep", ["start_post_comp", "start_post_init", "end_post_comp"])
def test_dse_run_does_not_support_dependencies(
    slurm_system: SlurmSystem, dse_tr: TestRun, dep: str, caplog: pytest.LogCaptureFixture
) -> None:
    """
    DSE runs do not support dependencies.

    DSE engine re-uses BaseRunner by manually controlling test_run to execute. BaseRunner doesn't keep track of all jobs
    and their statuses, this information is not available between cases in a scenario or even between steps of a single
    test run.

    While it might be useful in future, today we have to explicitly forbid such configurations and report actionable
    error to users.
    """
    dse_tr.dependencies = {dep: TestDependency(test_run=dse_tr)}
    test_scenario: TestScenario = TestScenario(name="test_scenario", test_runs=[dse_tr])
    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario)
    assert handle_dse_job(runner, argparse.Namespace(mode="dry-run")) == 1
    assert "Dependencies are not supported for DSE jobs, all cases run consecutively." in caplog.text
    assert "Please remove dependencies and re-run." in caplog.text
