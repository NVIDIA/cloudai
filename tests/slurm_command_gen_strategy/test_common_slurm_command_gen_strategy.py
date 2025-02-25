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
from typing import Dict, List, Optional, Union
from unittest.mock import Mock, create_autospec

import pytest

from cloudai import GitRepo, Test, TestRun, TestScenario, TestTemplate
from cloudai.systems import SlurmSystem
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition
from tests.conftest import create_autospec_dataclass


class MySlurmCommandGenStrategy(SlurmCommandGenStrategy):
    def _container_mounts(self, tr: TestRun) -> List[str]:
        return []


@pytest.fixture
def strategy_fixture(slurm_system: SlurmSystem) -> SlurmCommandGenStrategy:
    cmd_args: Dict[str, Union[str, List[str]]] = {"test_arg": "test_value"}
    strategy = MySlurmCommandGenStrategy(slurm_system, cmd_args)
    return strategy


@pytest.fixture
def testrun_fixture(tmp_path: Path) -> TestRun:
    tdef = NCCLTestDefinition(
        name="test_job",
        description="Test description",
        test_template_name="d",
        cmd_args=NCCLCmdArgs(),
        extra_env_vars={"TEST_VAR": "VALUE"},
    )
    mock_test_template = Mock(spec=TestTemplate)
    mock_test_template.name = "test_template"

    test = Test(test_definition=tdef, test_template=mock_test_template)

    return TestRun(name="test_job", test=test, output_path=tmp_path, num_nodes=2, nodes=["node1", "node2"])


def test_filename_generation(strategy_fixture: SlurmCommandGenStrategy, testrun_fixture: TestRun):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args: Dict[str, Union[str, List[str]]] = {"test_arg": "test_value"}

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, testrun_fixture)
    srun_command = strategy_fixture._gen_srun_command(slurm_args, env_vars, cmd_args, testrun_fixture)

    sbatch_command = strategy_fixture._write_sbatch_script(slurm_args, env_vars, srun_command, testrun_fixture)
    filepath_from_command = sbatch_command.split()[-1]

    assert testrun_fixture.output_path.joinpath("cloudai_sbatch_script.sh").exists()

    with open(filepath_from_command, "r") as file:
        file_contents = file.read()

    assert "test_job" in file_contents
    assert "node1,node2" in file_contents
    assert "srun" in file_contents
    assert "--mpi=fake-mpi" in file_contents


def test_num_nodes_and_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args: Dict[str, Union[str, List[str]]] = {"test_arg": "test_value"}
    tr = Mock(spec=TestRun)
    tr.nodes = ["node1", "node2"]
    tr.num_nodes = 3

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

    assert slurm_args["num_nodes"] == len(tr.nodes)


def test_only_num_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args: Dict[str, Union[str, List[str]]] = {"test_arg": "test_value"}
    tr = create_autospec(TestRun)
    tr.nodes = []
    tr.num_nodes = 3

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

    assert slurm_args["num_nodes"] == tr.num_nodes


def test_only_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args: Dict[str, Union[str, List[str]]] = {"test_arg": "test_value"}
    tr = create_autospec(TestRun)
    tr.num_nodes = 0
    tr.nodes = ["node1", "node2"]

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

    assert slurm_args["num_nodes"] == len(tr.nodes)


@pytest.mark.parametrize("time_limit", [None, "1:00:00"])
def test_time_limit(time_limit: Optional[str], strategy_fixture: SlurmCommandGenStrategy):
    tr = create_autospec_dataclass(TestRun)
    tr.nodes = []
    tr.time_limit = time_limit

    slurm_args = strategy_fixture._parse_slurm_args("prefix", {}, {}, tr)

    if time_limit is not None:
        assert slurm_args["time_limit"] == time_limit
    else:
        assert "time_limit" not in slurm_args


def test_raises_if_no_default_partition(slurm_system: SlurmSystem):
    slurm_system.default_partition = ""
    with pytest.raises(ValueError) as exc_info:
        MySlurmCommandGenStrategy(slurm_system, {})
    assert (
        "Default partition not set in the Slurm system object. "
        "The 'default_partition' attribute should be properly defined in the Slurm "
        "system configuration. Please ensure that 'default_partition' is set correctly "
        "in the corresponding system configuration (e.g., system.toml)."
    ) in str(exc_info.value)


@pytest.mark.parametrize(
    "pre_test,post_test,expected_script_lines",
    [
        # No pre_test, no post_test
        (None, None, ["srun"]),
        # One pre_test, no post_test
        (
            [Mock(test=Mock(name="test1", test_template=Mock()))],
            None,
            [
                "pre_test",
                "PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && echo 1 || echo 0 )",
                "if [ $PRE_TEST_SUCCESS -eq 1 ]; then",
                "    srun",
                "fi",
            ],
        ),
        # No pre_test, one post_test
        (
            None,
            [Mock(test=Mock(name="test2", test_template=Mock()))],
            [
                "srun",
                "post_test",
            ],
        ),
        # One pre_test, one post_test
        (
            [Mock(test=Mock(name="test1", test_template=Mock()))],
            [Mock(test=Mock(name="test2", test_template=Mock()))],
            [
                "pre_test",
                "PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && echo 1 || echo 0 )",
                "if [ $PRE_TEST_SUCCESS -eq 1 ]; then",
                "    srun",
                "    post_test",
                "fi",
            ],
        ),
        # Multiple pre_tests, multiple post_tests
        (
            [Mock(test=Mock(name="test1", test_template=Mock())), Mock(test=Mock(name="test2", test_template=Mock()))],
            [Mock(test=Mock(name="test3", test_template=Mock())), Mock(test=Mock(name="test4", test_template=Mock()))],
            [
                "pre_test",
                "pre_test",
                "PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && [ $SUCCESS_1 -eq 1 ] && echo 1 || echo 0 )",
                "if [ $PRE_TEST_SUCCESS -eq 1 ]; then",
                "    srun",
                "    post_test",
                "    post_test",
                "fi",
            ],
        ),
        # Multiple pre_tests, no post_test
        (
            [Mock(test=Mock(name="test1", test_template=Mock())), Mock(test=Mock(name="test2", test_template=Mock()))],
            None,
            [
                "pre_test",
                "pre_test",
                "PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && [ $SUCCESS_1 -eq 1 ] && echo 1 || echo 0 )",
                "if [ $PRE_TEST_SUCCESS -eq 1 ]; then",
                "    srun",
                "fi",
            ],
        ),
        # No pre_test, multiple post_tests
        (
            None,
            [Mock(test=Mock(name="test3", test_template=Mock())), Mock(test=Mock(name="test4", test_template=Mock()))],
            [
                "srun",
                "post_test",
                "post_test",
            ],
        ),
        # Multiple pre_tests, single post_test
        (
            [Mock(test=Mock(name="test1", test_template=Mock())), Mock(test=Mock(name="test2", test_template=Mock()))],
            [Mock(test=Mock(name="test3", test_template=Mock()))],
            [
                "pre_test",
                "pre_test",
                "PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && [ $SUCCESS_1 -eq 1 ] && echo 1 || echo 0 )",
                "if [ $PRE_TEST_SUCCESS -eq 1 ]; then",
                "    srun",
                "    post_test",
                "fi",
            ],
        ),
    ],
)
def test_pre_test_post_test_combinations(
    strategy_fixture: SlurmCommandGenStrategy,
    testrun_fixture: TestRun,
    pre_test,
    post_test,
    expected_script_lines,
):
    testrun_fixture.pre_test = Mock(spec=TestScenario) if pre_test else None
    testrun_fixture.post_test = Mock(spec=TestScenario) if post_test else None

    if pre_test is not None:
        testrun_fixture.pre_test = Mock(spec=TestScenario)
        testrun_fixture.pre_test.test_runs = pre_test
        for idx, run in enumerate(pre_test):
            run.test.test_template.gen_srun_success_check.return_value = "pre_test"
            run.test.test_template.gen_srun_command.return_value = "srun"
            run.test.name = f"test{idx+1}"

    if post_test is not None:
        testrun_fixture.post_test = Mock(spec=TestScenario)
        testrun_fixture.post_test.test_runs = post_test
        for idx, run in enumerate(post_test):
            run.test.test_template.gen_srun_command.return_value = "post_test"
            run.test.name = f"test{idx+1}"

    sbatch_command = strategy_fixture.gen_exec_command(testrun_fixture)
    script_file_path = sbatch_command.split()[-1]

    with open(script_file_path, "r") as script_file:
        script_content = script_file.read()

    for expected_line in expected_script_lines:
        assert expected_line in script_content, f"Expected '{expected_line}' in generated script but it was missing."


def test_default_container_mounts(strategy_fixture: SlurmCommandGenStrategy, testrun_fixture: TestRun):
    testrun_fixture.output_path = Path("./")
    mounts = strategy_fixture.container_mounts(testrun_fixture)
    assert len(mounts) == 1
    assert mounts[0] == f"{testrun_fixture.output_path.absolute()}:/cloudai_run_results"


def test_append_sbatch_directives(strategy_fixture: SlurmCommandGenStrategy, tmp_path: Path):
    content: list[str] = []
    strategy_fixture.system.extra_sbatch_args = ["--section=4", "--other-arg 1"]
    strategy_fixture._append_sbatch_directives(content, {"node_list_str": ""}, tmp_path)

    assert f"#SBATCH --partition={strategy_fixture.system.default_partition}" in content
    for arg in strategy_fixture.system.extra_sbatch_args:
        assert f"#SBATCH {arg}" in content


def test_default_container_mounts_with_extra_mounts(strategy_fixture: SlurmCommandGenStrategy):
    nccl = NCCLTestDefinition(
        name="name",
        description="desc",
        test_template_name="tt",
        cmd_args=NCCLCmdArgs(),
        extra_container_mounts=["/host:/container"],
    )
    t = Test(test_definition=nccl, test_template=Mock())
    tr = TestRun(name="t1", test=t, num_nodes=1, nodes=[], output_path=Path("./"))
    mounts = strategy_fixture.container_mounts(tr)
    assert len(mounts) == 2
    assert mounts[0] == f"{tr.output_path.absolute()}:/cloudai_run_results"
    assert mounts[1] == "/host:/container"


def test_default_container_mounts_with_git_repos(strategy_fixture: SlurmCommandGenStrategy):
    repo1 = GitRepo(url="./git_repo", commit="commit", mount_as="/git/r1", installed_path=Path.cwd())
    repo2 = GitRepo(url="./git_repo2", commit="commit", mount_as="/git/r2", installed_path=Path.cwd())
    nccl = NCCLTestDefinition(
        name="name",
        description="desc",
        test_template_name="tt",
        cmd_args=NCCLCmdArgs(),
        git_repos=[repo1, repo2],
    )
    t = Test(test_definition=nccl, test_template=Mock())
    tr = TestRun(name="t1", test=t, num_nodes=1, nodes=[], output_path=Path("./"))
    mounts = strategy_fixture.container_mounts(tr)
    assert len(mounts) == 3
    assert mounts[0] == f"{tr.output_path.absolute()}:/cloudai_run_results"
    assert mounts[1] == f"{repo1.installed_path}:{repo1.container_mount}"
    assert mounts[2] == f"{repo2.installed_path}:{repo2.container_mount}"
