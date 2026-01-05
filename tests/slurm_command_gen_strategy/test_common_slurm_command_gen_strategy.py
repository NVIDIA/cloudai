# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import List, Optional
from unittest.mock import Mock, patch

import pytest

from cloudai.core import GitRepo, TestRun, TestScenario
from cloudai.systems.slurm import SlurmCommandGenStrategy, SlurmSystem
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition


class MySlurmCommandGenStrategy(SlurmCommandGenStrategy):
    def _container_mounts(self) -> List[str]:
        return []


@pytest.fixture
def testrun_fixture(tmp_path: Path) -> TestRun:
    tdef = NCCLTestDefinition(
        name="test_job",
        description="Test description",
        test_template_name="d",
        cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl"),
        extra_env_vars={"TEST_VAR": "VALUE"},
    )

    return TestRun(name="test_job", test=tdef, output_path=tmp_path, num_nodes=2, nodes=[])


@pytest.fixture
def strategy_fixture(slurm_system: SlurmSystem, testrun_fixture: TestRun) -> SlurmCommandGenStrategy:
    strategy = MySlurmCommandGenStrategy(slurm_system, testrun_fixture)
    return strategy


def test_filename_generation(slurm_system: SlurmSystem, testrun_fixture: TestRun):
    testrun_fixture.nodes = ["node1", "node2"]
    strategy = MySlurmCommandGenStrategy(slurm_system, testrun_fixture)
    srun_command = strategy._gen_srun_command()

    sbatch_command = strategy._write_sbatch_script(srun_command)
    filepath_from_command = sbatch_command.split()[-1]

    assert strategy.test_run.output_path.joinpath("cloudai_sbatch_script.sh").exists()

    with open(filepath_from_command, "r") as file:
        file_contents = file.read()

    assert "node1,node2" in file_contents
    assert "srun" in file_contents
    assert f"--mpi={strategy.system.mpi}" in file_contents


def test_num_nodes_and_nodes(slurm_system: SlurmSystem):
    tr = make_test_run(name="test_job", output_dir=slurm_system.output_path)
    tr.nodes = ["node1", "node2"]
    tr.num_nodes = 3
    tr.output_path.mkdir(parents=True, exist_ok=True)
    strategy = MySlurmCommandGenStrategy(slurm_system, tr)

    lines: list[str] = []
    strategy._append_sbatch_directives(lines)

    assert f"#SBATCH -N {tr.num_nodes}" not in lines
    assert f"#SBATCH --nodelist={','.join(tr.nodes)}" in lines


def test_only_num_nodes(strategy_fixture: SlurmCommandGenStrategy):
    lines: list[str] = []
    strategy_fixture.test_run.nodes = []
    strategy_fixture.test_run.num_nodes = 3
    strategy_fixture._append_sbatch_directives(lines)

    assert f"#SBATCH -N {strategy_fixture.test_run.num_nodes}" in lines
    assert f"#SBATCH --nodelist={','.join(strategy_fixture.test_run.nodes)}" not in lines


def test_only_nodes(slurm_system: SlurmSystem):
    tr = make_test_run(name="test_job", output_dir=slurm_system.output_path)
    tr.num_nodes = 0
    tr.nodes = ["node1", "node2"]
    tr.output_path.mkdir(parents=True, exist_ok=True)

    strategy = MySlurmCommandGenStrategy(slurm_system, tr)

    lines: list[str] = []
    strategy._append_sbatch_directives(lines)

    assert f"#SBATCH --nodelist={','.join(tr.nodes)}" in lines
    assert f"#SBATCH -N {tr.num_nodes}" not in lines


@pytest.mark.parametrize("time_limit", [None, "1:00:00"])
def test_time_limit(time_limit: Optional[str], strategy_fixture: SlurmCommandGenStrategy):
    lines: list[str] = []
    strategy_fixture.test_run.nodes = []
    strategy_fixture.test_run.time_limit = time_limit
    strategy_fixture._append_sbatch_directives(lines)

    if time_limit is not None:
        assert any("#SBATCH --time=" in line for line in lines)
    else:
        assert not any("#SBATCH --time=" in line for line in lines)


def make_test_run(name: str, output_dir: Path) -> TestRun:
    test_def = NCCLTestDefinition(
        name=name,
        description=name,
        test_template_name="nccl",
        cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl"),
        extra_env_vars={"TEST_VAR": "VALUE"},
    )
    return TestRun(name=name, test=test_def, num_nodes=1, nodes=["node1"], output_path=output_dir / name)


def test_pre_post_combinations(tmp_path: Path, slurm_system: SlurmSystem, strategy_fixture: SlurmCommandGenStrategy):
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
            pre_runs = [make_test_run(f"pre{i}", tmp_path) for i in range(pre_count)]
            strategy_fixture.test_run.pre_test = TestScenario(name="pre", test_runs=pre_runs)
        else:
            strategy_fixture.test_run.pre_test = None

        if post_count > 0:
            post_runs = [make_test_run(f"post{i}", tmp_path) for i in range(post_count)]
            strategy_fixture.test_run.post_test = TestScenario(name="post", test_runs=post_runs)
        else:
            strategy_fixture.test_run.post_test = None

        sbatch_command = strategy_fixture.gen_exec_command()
        script_path = sbatch_command.split()[-1]
        content = Path(script_path).read_text()

        for expected in expected_lines:
            assert any(expected in line for line in content.splitlines()), f"Missing line: {expected}"


def test_default_container_mounts(strategy_fixture: SlurmCommandGenStrategy):
    strategy_fixture.test_run.output_path = Path("./")
    mounts = strategy_fixture.container_mounts()
    assert len(mounts) == 3
    assert mounts[0] == f"{strategy_fixture.test_run.output_path.absolute()}:/cloudai_run_results"
    assert mounts[1] == f"{strategy_fixture.system.install_path.absolute()}:/cloudai_install"
    assert mounts[2] == f"{strategy_fixture.test_run.output_path.absolute()}"


def test_append_sbatch_directives(strategy_fixture: SlurmCommandGenStrategy):
    content: list[str] = []
    strategy_fixture.system.extra_sbatch_args = ["--section=4", "--other-arg 1"]
    strategy_fixture._append_sbatch_directives(content)

    assert f"#SBATCH --partition={strategy_fixture.system.default_partition}" in content
    for arg in strategy_fixture.system.extra_sbatch_args:
        assert f"#SBATCH {arg}" in content


def test_default_container_mounts_with_extra_mounts(strategy_fixture: SlurmCommandGenStrategy):
    strategy_fixture.test_run.test.extra_container_mounts = ["/host:/container"]
    mounts = strategy_fixture.container_mounts()
    assert len(mounts) == 4
    assert mounts[0] == f"{strategy_fixture.test_run.output_path.absolute()}:/cloudai_run_results"
    assert mounts[1] == f"{strategy_fixture.system.install_path.absolute()}:/cloudai_install"
    assert mounts[2] == f"{strategy_fixture.test_run.output_path.absolute()}"
    assert mounts[3] == "/host:/container"


def test_default_container_mounts_with_git_repos(strategy_fixture: SlurmCommandGenStrategy):
    repo1 = GitRepo(url="./git_repo", commit="commit", mount_as="/git/r1", installed_path=Path.cwd())
    repo2 = GitRepo(url="./git_repo2", commit="commit", mount_as="/git/r2", installed_path=Path.cwd())
    strategy_fixture.test_run.test.git_repos = [repo1, repo2]

    mounts = strategy_fixture.container_mounts()

    assert len(mounts) == 5
    assert mounts[0] == f"{strategy_fixture.test_run.output_path.absolute()}:/cloudai_run_results"
    assert mounts[1] == f"{strategy_fixture.system.install_path.absolute()}:/cloudai_install"
    assert mounts[2] == f"{strategy_fixture.test_run.output_path.absolute()}"
    assert mounts[3] == f"{repo1.installed_path}:{repo1.container_mount}"
    assert mounts[4] == f"{repo2.installed_path}:{repo2.container_mount}"


def test_ranks_mapping_cmd(strategy_fixture: SlurmCommandGenStrategy):
    expected_command = (
        f"srun --export=ALL --mpi={strategy_fixture.system.mpi} -N{strategy_fixture.test_run.num_nodes} "
        f"--output={strategy_fixture.test_run.output_path.absolute()}/mapping-stdout.txt "
        f"--error={strategy_fixture.test_run.output_path.absolute()}/mapping-stderr.txt "
        "bash -c "
        r'"echo \$(date): \$(hostname):node \${SLURM_NODEID}:rank \${SLURM_PROCID}."'
    )

    result = strategy_fixture._ranks_mapping_cmd()
    assert result == expected_command


def test_nccl_topo_mount(strategy_fixture: SlurmCommandGenStrategy):
    strategy_fixture.test_run.test.extra_env_vars["NCCL_TOPO_FILE"] = "/tmp/nccl_topo.txt"
    mounts = strategy_fixture.container_mounts()
    expected_mount = f"{Path('/tmp/nccl_topo.txt').resolve()}:{Path('/tmp/nccl_topo.txt').resolve()}"
    assert expected_mount in mounts


@pytest.mark.parametrize("use_pretest_extras", [True, False])
def test_gen_srun_prefix_with_pretest_extras(
    use_pretest_extras: bool, tmp_path: Path, slurm_system: SlurmSystem, strategy_fixture: SlurmCommandGenStrategy
):
    pre_test = TestScenario(name="pre", test_runs=[make_test_run("pre", tmp_path)])

    class PreTestCmdGenStrategy(MySlurmCommandGenStrategy):
        def pre_test_srun_extra_args(self, tr: TestRun) -> List[str]:
            return ["--pre-arg1", "--pre-arg2"]

    strategy_fixture.test_run.pre_test = pre_test
    with patch.object(
        strategy_fixture,
        "_get_cmd_gen_strategy",
        return_value=PreTestCmdGenStrategy(slurm_system, strategy_fixture.test_run),
    ):
        srun_prefix_with_extras = strategy_fixture.gen_srun_prefix(use_pretest_extras=use_pretest_extras)

    assert ("--pre-arg1" in srun_prefix_with_extras) is use_pretest_extras
    assert ("--pre-arg2" in srun_prefix_with_extras) is use_pretest_extras


@pytest.mark.parametrize("container_mount_home", [True, False])
def test_gen_srun_prefix_with_container_mount_home(
    container_mount_home: bool, strategy_fixture: SlurmCommandGenStrategy
):
    strategy_fixture.system.container_mount_home = container_mount_home
    strategy_fixture.image_path = Mock(return_value="path")
    srun_prefix = strategy_fixture.gen_srun_prefix()
    assert ("--no-container-mount-home" in srun_prefix) is not container_mount_home


def test_gen_srun_prefix_tr_extra_srun_args(strategy_fixture: SlurmCommandGenStrategy):
    strategy_fixture.test_run.extra_srun_args = "--arg val --flag"
    srun_prefix = strategy_fixture.gen_srun_prefix()
    assert "--arg val --flag" in srun_prefix  # added as a single element


def test_append_distribution_and_hostfile_with_nodes(slurm_system: SlurmSystem, testrun_fixture: TestRun) -> None:
    slurm_system.distribution = "block"
    slurm_system.ntasks_per_node = 2
    testrun_fixture.nodes = ["node1", "node2"]
    strategy = MySlurmCommandGenStrategy(slurm_system, testrun_fixture)
    content: List[str] = []
    strategy._append_nodes_related_directives(content)

    assert "#SBATCH --distribution=block" in content
    assert "#SBATCH --nodelist=node1,node2" in content

    hostfile_path = strategy.test_run.output_path / "hostfile.txt"
    assert hostfile_path.exists()
    lines: List[str] = hostfile_path.read_text().splitlines()
    assert lines == ["node1", "node1", "node2", "node2"]


def test_distribution_fallback_when_no_nodes(strategy_fixture: SlurmCommandGenStrategy) -> None:
    strategy_fixture.test_run.nodes = []
    strategy_fixture.system.distribution = "cyclic"
    content: List[str] = []
    strategy_fixture._append_nodes_related_directives(content)

    assert "#SBATCH --distribution=cyclic" in content
    assert "#SBATCH --nodelist=" not in content


def test_nodelist_over_num_nodes(slurm_system: SlurmSystem, testrun_fixture: TestRun) -> None:
    testrun_fixture.nodes = ["nodeA", "nodeB", "nodeC"]
    testrun_fixture.num_nodes = 5
    strategy = MySlurmCommandGenStrategy(slurm_system, testrun_fixture)
    content: list[str] = []

    strategy._append_sbatch_directives(content)
    assert "#SBATCH --nodelist=nodeA,nodeB,nodeC" in content
    assert "#SBATCH -N" not in content

    cmd = strategy.gen_srun_prefix(with_num_nodes=True)
    assert " -N" not in " ".join(cmd)
