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

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import Mock, create_autospec

import pytest

from cloudai import GitRepo, Test, TestRun, TestScenario, TestTemplate
from cloudai.systems import SlurmSystem
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition, NcclTestSlurmCommandGenStrategy
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
    env_vars: Dict[str, Union[str, List[str]]] = {"TEST_VAR": "VALUE"}
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
    assert f"--mpi={strategy_fixture.system.mpi}" in file_contents


def test_num_nodes_and_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars: Dict[str, Union[str, List[str]]] = {"TEST_VAR": "VALUE"}
    cmd_args: Dict[str, Union[str, List[str]]] = {"test_arg": "test_value"}
    tr = Mock(spec=TestRun)
    tr.nodes = ["node1", "node2"]
    tr.num_nodes = 3

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

    assert slurm_args["num_nodes"] == len(tr.nodes)


def test_only_num_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars: Dict[str, Union[str, List[str]]] = {"TEST_VAR": "VALUE"}
    cmd_args: Dict[str, Union[str, List[str]]] = {"test_arg": "test_value"}
    tr = create_autospec(TestRun)
    tr.nodes = []
    tr.num_nodes = 3

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

    assert slurm_args["num_nodes"] == tr.num_nodes


def test_only_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars: Dict[str, Union[str, List[str]]] = {"TEST_VAR": "VALUE"}
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


def make_test_run(slurm_system: SlurmSystem, name: str, output_dir: Path) -> TestRun:
    test_def = NCCLTestDefinition(
        name=name,
        description=name,
        test_template_name="nccl",
        cmd_args=NCCLCmdArgs(),
        extra_env_vars={"TEST_VAR": "VALUE"},
    )
    test_template = TestTemplate(slurm_system, "nccl")
    test_template.command_gen_strategy = NcclTestSlurmCommandGenStrategy(slurm_system, test_def.cmd_args_dict)
    test = Test(test_definition=test_def, test_template=test_template)
    return TestRun(name=name, test=test, num_nodes=1, nodes=["node1"], output_path=output_dir / name)


def test_pre_post_combinations(
    tmp_path: Path,
    slurm_system: SlurmSystem,
    strategy_fixture: SlurmCommandGenStrategy,
    testrun_fixture: TestRun,
):
    test_cases = [
        {
            "pre_count": 0,
            "post_count": 0,
            "expected_lines": ["srun --export=ALL --mpi="],
        },
        {
            "pre_count": 1,
            "post_count": 0,
            "expected_lines": [
                "PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ]",
                "if [ $PRE_TEST_SUCCESS -eq 1 ]; then",
                "srun --export=ALL --mpi=",
                "fi",
            ],
        },
        {
            "pre_count": 0,
            "post_count": 1,
            "expected_lines": [
                "srun --export=ALL --mpi=",
                "/post_test/",
            ],
        },
        {
            "pre_count": 1,
            "post_count": 1,
            "expected_lines": [
                "PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ]",
                "if [ $PRE_TEST_SUCCESS -eq 1 ]; then",
                "srun --export=ALL --mpi=",
                "/post_test/",
                "fi",
            ],
        },
        {
            "pre_count": 2,
            "post_count": 2,
            "expected_lines": [
                "PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && [ $SUCCESS_1 -eq 1 ]",
                "if [ $PRE_TEST_SUCCESS -eq 1 ]; then",
                "srun --export=ALL --mpi=",
                "/post_test/",
            ],
        },
        {
            "pre_count": 2,
            "post_count": 0,
            "expected_lines": [
                "PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && [ $SUCCESS_1 -eq 1 ]",
                "if [ $PRE_TEST_SUCCESS -eq 1 ]; then",
                "srun --export=ALL --mpi=",
                "fi",
            ],
        },
        {
            "pre_count": 0,
            "post_count": 2,
            "expected_lines": [
                "srun --export=ALL --mpi=",
                "/post_test/",
            ],
        },
        {
            "pre_count": 2,
            "post_count": 1,
            "expected_lines": [
                "PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && [ $SUCCESS_1 -eq 1 ]",
                "if [ $PRE_TEST_SUCCESS -eq 1 ]; then",
                "srun --export=ALL --mpi=",
                "/post_test/",
                "fi",
            ],
        },
    ]

    for case in test_cases:
        pre_count = case["pre_count"]
        post_count = case["post_count"]
        expected_lines = case["expected_lines"]

        if pre_count > 0:
            pre_runs = [make_test_run(slurm_system, f"pre{i}", tmp_path) for i in range(pre_count)]
            testrun_fixture.pre_test = TestScenario(name="pre", test_runs=pre_runs)
        else:
            testrun_fixture.pre_test = None

        if post_count > 0:
            post_runs = [make_test_run(slurm_system, f"post{i}", tmp_path) for i in range(post_count)]
            testrun_fixture.post_test = TestScenario(name="post", test_runs=post_runs)
        else:
            testrun_fixture.post_test = None

        sbatch_command = strategy_fixture.gen_exec_command(testrun_fixture)
        script_path = sbatch_command.split()[-1]
        content = Path(script_path).read_text()

        for expected in expected_lines:
            assert any(expected in line for line in content.splitlines()), f"Missing line: {expected}"


def test_default_container_mounts(strategy_fixture: SlurmCommandGenStrategy, testrun_fixture: TestRun):
    testrun_fixture.output_path = Path("./")
    mounts = strategy_fixture.container_mounts(testrun_fixture)
    assert len(mounts) == 2
    assert mounts[0] == f"{testrun_fixture.output_path.absolute()}:/cloudai_run_results"
    assert mounts[1] == f"{strategy_fixture.system.install_path.absolute()}:/cloudai_install"


def test_append_sbatch_directives(strategy_fixture: SlurmCommandGenStrategy, testrun_fixture: TestRun):
    content: list[str] = []
    strategy_fixture.system.extra_sbatch_args = ["--section=4", "--other-arg 1"]
    strategy_fixture._append_sbatch_directives(content, {"node_list_str": ""}, testrun_fixture)

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
    assert len(mounts) == 3
    assert mounts[0] == f"{tr.output_path.absolute()}:/cloudai_run_results"
    assert mounts[1] == "/host:/container"
    assert mounts[2] == f"{strategy_fixture.system.install_path.absolute()}:/cloudai_install"


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
    assert len(mounts) == 4
    assert mounts[0] == f"{tr.output_path.absolute()}:/cloudai_run_results"
    assert mounts[1] == f"{repo1.installed_path}:{repo1.container_mount}"
    assert mounts[2] == f"{repo2.installed_path}:{repo2.container_mount}"
    assert mounts[3] == f"{strategy_fixture.system.install_path.absolute()}:/cloudai_install"


def test_ranks_mapping_cmd(strategy_fixture: SlurmCommandGenStrategy, testrun_fixture: TestRun):
    slurm_args = {"job_name": "test_job", "num_nodes": 2, "node_list_str": "node1,node2"}

    expected_command = (
        f"srun --export=ALL --mpi={strategy_fixture.system.mpi} "
        f"--output={testrun_fixture.output_path.absolute()}/mapping-stdout.txt "
        f"--error={testrun_fixture.output_path.absolute()}/mapping-stderr.txt "
        "bash -c "
        r'"echo \$(date): \$(hostname):node \${SLURM_NODEID}:rank \${SLURM_PROCID}."'
    )

    result = strategy_fixture._ranks_mapping_cmd(slurm_args, testrun_fixture)
    assert result == expected_command


def test_nccl_topo_mount(strategy_fixture: SlurmCommandGenStrategy, testrun_fixture: TestRun):
    testrun_fixture.test.extra_env_vars["NCCL_TOPO_FILE"] = "/tmp/nccl_topo.txt"
    mounts = strategy_fixture.container_mounts(testrun_fixture)
    expected_mount = f"{Path('/tmp/nccl_topo.txt').resolve()}:{Path('/tmp/nccl_topo.txt').resolve()}"
    assert expected_mount in mounts


def test_append_distribution_and_hostfile_with_nodes(
    strategy_fixture: SlurmCommandGenStrategy, testrun_fixture: TestRun
) -> None:
    strategy_fixture.system.distribution = "block"
    strategy_fixture.system.ntasks_per_node = 2
    content: List[str] = []
    args: Dict[str, Any] = {"node_list_str": ",".join(testrun_fixture.nodes)}
    strategy_fixture._append_distribution_and_hostfile(content, args, testrun_fixture)

    assert "#SBATCH --distribution=arbitrary" in content

    hostfile_line: str = next(line for line in content if line.startswith("export SLURM_HOSTFILE="))
    hostfile_path: Path = Path(hostfile_line.split("=", 1)[1])
    assert hostfile_path.exists()
    lines: List[str] = hostfile_path.read_text().splitlines()
    assert lines == ["node1", "node1", "node2", "node2"]


def test_distribution_fallback_when_no_nodes(
    strategy_fixture: SlurmCommandGenStrategy, testrun_fixture: TestRun
) -> None:
    testrun_fixture.nodes = []
    strategy_fixture.system.distribution = "cyclic"
    content: List[str] = []
    args: Dict[str, Any] = {"node_list_str": ""}
    strategy_fixture._append_distribution_and_hostfile(content, args, testrun_fixture)

    assert "#SBATCH --distribution=cyclic" in content
