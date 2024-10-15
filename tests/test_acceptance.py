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

import argparse
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pytest

from cloudai import Test
from cloudai.cli import setup_logging
from cloudai.cli.handlers import handle_dry_run_and_run, identify_unique_test_templates
from cloudai.schema.test_template.nccl_test.template import NcclTest
from cloudai.schema.test_template.ucc_test.template import UCCTest
from cloudai.systems.standalone_system import StandaloneSystem
from cloudai.test_definitions.nccl import NCCLCmdArgs, NCCLTestDefinition

SLURM_TEST_SCENARIOS = [
    {"path": Path("conf/common/test_scenario/sleep.toml"), "expected_dirs_number": 4, "log_file": "sleep_debug.log"},
    {
        "path": Path("conf/common/test_scenario/ucc_test.toml"),
        "expected_dirs_number": 5,
        "log_file": "ucc_test_debug.log",
    },
]


@pytest.mark.parametrize("scenario", SLURM_TEST_SCENARIOS, ids=lambda x: str(x))
def test_slurm(tmp_path: Path, scenario: Dict):
    test_scenario_path = scenario["path"]
    expected_dirs_number = scenario.get("expected_dirs_number")
    log_file = scenario.get("log_file", ".")
    log_file_path = tmp_path / log_file

    setup_logging(log_file_path, "DEBUG")
    args = argparse.Namespace(
        mode="dry-run",
        system_config=Path("conf/common/system/example_slurm_cluster.toml"),
        test_templates_dir=Path("conf/common/test_template"),
        tests_dir=Path("conf/common/test"),
        test_scenario=test_scenario_path,
        output_dir=tmp_path,
    )
    with patch("asyncio.sleep", return_value=None):
        handle_dry_run_and_run(args)

    # Find the directory that was created for the test results
    results_output_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]

    # Assuming there's only one result directory created
    assert len(results_output_dirs) == 1, "No result directory found or multiple directories found."
    results_output = results_output_dirs[0]

    test_dirs = list(results_output.iterdir())

    if expected_dirs_number is not None:
        assert len(test_dirs) == expected_dirs_number, "Dirs number in output is not as expected"

    for td in test_dirs:
        assert td.is_dir(), "Invalid test directory"
        assert "Tests." in td.name, "Invalid test directory name"

    assert log_file_path.exists(), f"Log file {log_file_path} was not created"


class TestIdentifyUniqueTestTemplates:
    @pytest.fixture
    def system(self, tmp_path: Path) -> StandaloneSystem:
        return StandaloneSystem(name="system", install_path=tmp_path, output_path=tmp_path)

    @pytest.fixture
    def test_def(self) -> NCCLTestDefinition:
        return NCCLTestDefinition(name="nccl", description="", test_template_name="ttname", cmd_args=NCCLCmdArgs())

    def test_single_input(self, system: StandaloneSystem, test_def: NCCLTestDefinition):
        templ = NcclTest(system, "template_name")
        test = Test(test_definition=test_def, test_template=templ)

        res = identify_unique_test_templates([test])

        assert len(res) == 1
        assert res[0] == templ

    def test_two_templates_with_different_names(self, system: StandaloneSystem, test_def: NCCLTestDefinition):
        templ1 = NcclTest(system, "template_name1")
        templ2 = NcclTest(system, "template_name2")
        test1 = Test(test_definition=test_def, test_template=templ1)
        test2 = Test(test_definition=test_def, test_template=templ2)

        res = identify_unique_test_templates([test1, test2])

        assert len(res) == 1
        assert res[0] == templ1

    def test_two_templates_with_same_name(self, system: StandaloneSystem, test_def: NCCLTestDefinition):
        templ = NcclTest(system, "template_name")
        test1 = Test(test_definition=test_def, test_template=templ)
        test2 = Test(test_definition=test_def, test_template=templ)

        res = identify_unique_test_templates([test1, test2])

        assert len(res) == 1
        assert res[0] == templ

    def test_two_different_templates_with_same_name(self, system: StandaloneSystem, test_def: NCCLTestDefinition):
        templ1 = NcclTest(system, "template_name")
        templ2 = UCCTest(system, "template_name")
        test1 = Test(test_definition=test_def, test_template=templ1)
        test2 = Test(test_definition=test_def, test_template=templ2)

        res = identify_unique_test_templates([test1, test2])

        assert len(res) == 2
        assert templ1 in res
        assert templ2 in res
