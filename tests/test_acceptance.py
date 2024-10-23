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
from concurrent.futures import Future
from functools import partial
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, Mock, patch

import pytest
from cloudai import BaseInstaller, InstallStatusResult, NcclTest, Test, TestRun, TestTemplate, UCCTest
from cloudai.cli import handle_dry_run_and_run, identify_unique_test_templates, setup_logging
from cloudai.schema.test_template.nccl_test.slurm_command_gen_strategy import NcclTestSlurmCommandGenStrategy
from cloudai.schema.test_template.sleep.slurm_command_gen_strategy import SleepSlurmCommandGenStrategy
from cloudai.schema.test_template.sleep.template import Sleep
from cloudai.schema.test_template.ucc_test.slurm_command_gen_strategy import UCCTestSlurmCommandGenStrategy
from cloudai.systems import SlurmSystem, StandaloneSystem
from cloudai.test_definitions.nccl import NCCLCmdArgs, NCCLTestDefinition
from cloudai.test_definitions.sleep import SleepCmdArgs, SleepTestDefinition
from cloudai.test_definitions.ucc import UCCCmdArgs, UCCTestDefinition

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


@pytest.fixture
def test_template_success() -> TestTemplate:
    template = MagicMock(spec=TestTemplate)
    template.name = "test_template_success"
    template.install.return_value = InstallStatusResult(success=True)
    template.uninstall.return_value = InstallStatusResult(success=True)
    return template


@pytest.fixture
def test_template_failure() -> TestTemplate:
    template = MagicMock(spec=TestTemplate)
    template.name = "test_template_failure"
    template.install.return_value = InstallStatusResult(success=False, message="Installation failed")
    template.uninstall.return_value = InstallStatusResult(success=False, message="Uninstallation failed")
    return template


def create_real_future(result):
    future = Future()
    future.set_result(result)
    return future


def extract_unique_test_templates(test_templates):
    unique_test_templates = {}
    for test_template in test_templates:
        template_name = test_template.name
        if template_name not in unique_test_templates:
            unique_test_templates[template_name] = test_template
    return list(unique_test_templates.values())


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_install_success(mock_executor: Mock, slurm_system: SlurmSystem, test_template_success: Mock):
    installer = BaseInstaller(slurm_system)
    mock_future = create_real_future(test_template_success.install.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.install([test_template_success])

    assert result.success
    assert result.message == "All test templates installed successfully."

    # Check if the template is installed
    assert installer.is_installed([test_template_success])


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_install_failure(mock_executor: Mock, slurm_system: SlurmSystem, test_template_failure: Mock):
    installer = BaseInstaller(slurm_system)
    mock_future = create_real_future(test_template_failure.install.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.install([test_template_failure])

    assert not result.success
    assert result.message == "Some test templates failed to install."


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_uninstall_success(mock_executor: Mock, slurm_system: SlurmSystem, test_template_success: Mock):
    installer = BaseInstaller(slurm_system)
    mock_future = create_real_future(test_template_success.uninstall.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.uninstall([test_template_success])

    assert result.success
    assert result.message == "All test templates uninstalled successfully."


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_uninstall_failure(mock_executor: Mock, slurm_system: SlurmSystem, test_template_failure: Mock):
    installer = BaseInstaller(slurm_system)
    mock_future = create_real_future(test_template_failure.uninstall.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.uninstall([test_template_failure])

    assert not result.success
    assert result.message == "Some test templates failed to uninstall."

    # Check if the template is still installed
    assert installer.is_installed([test_template_failure])


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


@pytest.fixture
def partial_tr(slurm_system: SlurmSystem) -> partial[TestRun]:
    return partial(TestRun, num_nodes=1, nodes=[], output_path=slurm_system.output_path)


@pytest.fixture(params=["ucc", "nccl", "sleep"])
def test_req(request, slurm_system: SlurmSystem, partial_tr: partial[TestRun]) -> tuple[TestRun, str]:
    if request.param == "ucc":
        tr = partial_tr(
            name="ucc",
            test=Test(
                test_definition=UCCTestDefinition(
                    name="ucc", description="ucc", test_template_name="ucc", cmd_args=UCCCmdArgs()
                ),
                test_template=UCCTest(slurm_system, name="ucc"),
            ),
        )
        tr.test.test_template.command_gen_strategy = UCCTestSlurmCommandGenStrategy(
            slurm_system, tr.test.test_definition.cmd_args_dict
        )
        tr.test.test_template.command_gen_strategy.job_name = Mock(return_value="job_name")

        return (tr, "ucc.sbatch")
    elif request.param == "nccl":
        tr = partial_tr(
            name="nccl",
            test=Test(
                test_definition=NCCLTestDefinition(
                    name="nccl", description="nccl", test_template_name="nccl", cmd_args=NCCLCmdArgs()
                ),
                test_template=NcclTest(slurm_system, name="nccl"),
            ),
        )
        tr.test.test_template.command_gen_strategy = NcclTestSlurmCommandGenStrategy(
            slurm_system, tr.test.test_definition.cmd_args_dict
        )
        tr.test.test_template.command_gen_strategy.job_name = Mock(return_value="job_name")

        return (tr, "nccl.sbatch")
    elif request.param == "sleep":
        tr = partial_tr(
            name="sleep",
            test=Test(
                test_definition=SleepTestDefinition(
                    name="sleep", description="sleep", test_template_name="sleep", cmd_args=SleepCmdArgs()
                ),
                test_template=Sleep(slurm_system, name="sleep"),
            ),
        )
        tr.test.test_template.command_gen_strategy = SleepSlurmCommandGenStrategy(
            slurm_system, tr.test.test_definition.cmd_args_dict
        )
        tr.test.test_template.command_gen_strategy.job_name = Mock(return_value="job_name")

        return (tr, "sleep.sbatch")

    raise ValueError(f"Unknown test: {request.param}")


def test_sbatch_generation(slurm_system: SlurmSystem, test_req: tuple[TestRun, str]):
    slurm_system.output_path.mkdir(parents=True, exist_ok=True)

    tr = test_req[0]

    sbatch_script = tr.test.test_template.gen_exec_command(tr).split()[-1]

    curr = Path(sbatch_script).read_text()
    ref = (Path(__file__).parent / "ref_data" / test_req[1]).read_text()
    ref = ref.replace("__OUTPUT_DIR__", str(slurm_system.output_path)).replace("__JOB_NAME__", "job_name")

    assert curr == ref
