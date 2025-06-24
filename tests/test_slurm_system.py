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

import re
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch

import pytest
import toml

from cloudai.core import BaseJob, Test, TestRun, TestTemplate
from cloudai.models.scenario import ReportConfig
from cloudai.systems.slurm import (
    SlurmCommandGenStrategy,
    SlurmNode,
    SlurmNodeState,
    SlurmSystem,
    parse_node_list,
)
from cloudai.systems.slurm.slurm_metadata import SlurmStepMetadata
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition


@pytest.mark.parametrize(
    "squeue_output,expected_nodes",
    [
        ("", []),
        ("partition|RUNNING|nodeA[001-002]|user", ["nodeA001", "nodeA002"]),
        (
            "p|RUNNING|nodeA[001-002]|user\np|RUNNING|nodeA[003-004]|user",
            ["nodeA001", "nodeA002", "nodeA003", "nodeA004"],
        ),
        ("partition|||user", []),  # nodes were not assigned yet
    ],
)
def test_nodes_from_squeue(slurm_system: SlurmSystem, squeue_output: str, expected_nodes: list[str]):
    with patch("cloudai.systems.slurm.SlurmSystem.fetch_command_output", return_value=(squeue_output, "")):
        nodes = slurm_system.nodes_from_squeue()
        assert [n.name for n in nodes] == expected_nodes
        assert all(n.state == SlurmNodeState.ALLOCATED for n in nodes)
        assert all(n.user == "user" for n in nodes)


def test_nodes_from_sinfo(slurm_system: SlurmSystem) -> None:
    sinfo_output = """main|inval||node-036
    main|drain||node-[045-046,059,061-062]
    main|resv||node-[034-035]
    main|alloc|user|node-[033,037-044,047-058,060,063-064]
    backup|idle|Unknown|node[01-08]
    empty_queue|||
    invalid_line
    """
    with patch("cloudai.systems.slurm.SlurmSystem.fetch_command_output", return_value=(sinfo_output, "")):
        all_nodes = slurm_system.nodes_from_sinfo()

    all_nodes_set = set(all_nodes)

    specs = {
        "inval": ("node-036", SlurmNodeState.INVALID_REGISTRATION, "", "main"),
        "drain": ("node-[045-046,059,061-062]", SlurmNodeState.DRAINED, "", "main"),
        "resv": ("node-[034-035]", SlurmNodeState.RESERVED, "", "main"),
        "alloc": ("node-[033,037-044,047-058,060,063-064]", SlurmNodeState.ALLOCATED, "user", "main"),
        "idle": ("node[01-08]", SlurmNodeState.IDLE, "Unknown", "backup"),
    }
    node_count = 0
    diff = all_nodes_set
    for key, (nodelist, state, user, partition) in specs.items():
        nodes = [
            SlurmNode(name=name, partition=partition, state=state, user=user) for name in parse_node_list(nodelist)
        ]
        node_count += len(nodes)
        for node in nodes:
            assert node in all_nodes_set, f"{key} node {node} not found in nodes"
        diff -= set(nodes)

    assert len(diff) == 0, f"Nodes not found in specs: {diff}"
    assert len(all_nodes) == node_count, f"Parsed {len(all_nodes)} nodes, expected {node_count}"


def test_update_nodes_state_and_user(slurm_system: SlurmSystem) -> None:
    for partition in slurm_system.partitions:
        partition.slurm_nodes.clear()

    assert len(slurm_system.partitions) == 2
    assert len(slurm_system.partitions[0].slurm_nodes) == 0
    assert len(slurm_system.partitions[1].slurm_nodes) == 0

    # insert new is False by default
    node = SlurmNode(name="node-036", partition="main", state=SlurmNodeState.IDLE)
    slurm_system.update_nodes_state_and_user([node])
    assert len(slurm_system.partitions[0].slurm_nodes) == 0
    assert len(slurm_system.partitions[1].slurm_nodes) == 0

    slurm_system.update_nodes_state_and_user([node], insert_new=True)
    assert len(slurm_system.partitions[0].slurm_nodes) == 1
    assert slurm_system.partitions[0].slurm_nodes[0] == node

    # existing is updated by default
    upd_node = SlurmNode(name="node-036", partition="main", state=SlurmNodeState.DRAINED)
    assert slurm_system.partitions[0].slurm_nodes[0].state != upd_node.state
    slurm_system.update_nodes_state_and_user([upd_node])
    assert slurm_system.partitions[0].slurm_nodes[0].state == upd_node.state

    # update non-default partition, previous state is preserved
    node2 = SlurmNode(name="node-backup-001", partition="backup", state=SlurmNodeState.IDLE)
    slurm_system.update_nodes_state_and_user([node2], insert_new=True)
    assert len(slurm_system.partitions[0].slurm_nodes) == 1
    assert slurm_system.partitions[0].slurm_nodes[0] == node
    assert len(slurm_system.partitions[1].slurm_nodes) == 1
    assert slurm_system.partitions[1].slurm_nodes[0] == node2


@pytest.fixture
def grouped_nodes() -> dict[SlurmNodeState, list[SlurmNode]]:
    """
    Helper function to set up a mock Slurm system with nodes and their states.
    """
    partition_name = "main"

    grouped_nodes = {
        SlurmNodeState.IDLE: [
            SlurmNode(name="node01", partition=partition_name, state=SlurmNodeState.IDLE),
            SlurmNode(name="node02", partition=partition_name, state=SlurmNodeState.IDLE),
        ],
        SlurmNodeState.COMPLETING: [
            SlurmNode(name="node04", partition=partition_name, state=SlurmNodeState.COMPLETING)
        ],
        SlurmNodeState.ALLOCATED: [SlurmNode(name="node05", partition=partition_name, state=SlurmNodeState.ALLOCATED)],
    }

    return grouped_nodes


def test_get_available_nodes_exceeding_limit_no_callstack(
    slurm_system: SlurmSystem, grouped_nodes: Dict[SlurmNodeState, List[SlurmNode]], caplog
):
    group_name = "group1"
    partition_name = "main"
    num_nodes = 5

    slurm_system.get_available_nodes_from_group(partition_name, group_name, num_nodes)

    log_message = "CloudAI is requesting 5 nodes from the group 'group1', but only 0 nodes are available."
    assert log_message in caplog.text


def test_allocate_nodes_max_avail(slurm_system: SlurmSystem, grouped_nodes: dict[SlurmNodeState, list[SlurmNode]]):
    group_name = "group_name"

    available_nodes = slurm_system.allocate_nodes(grouped_nodes, "max_avail", group_name)
    expected_node_names = [
        grouped_nodes[SlurmNodeState.IDLE][0].name,
        grouped_nodes[SlurmNodeState.IDLE][1].name,
        grouped_nodes[SlurmNodeState.COMPLETING][0].name,
    ]
    returned_node_names = [node.name for node in available_nodes]

    assert set(returned_node_names) == set(expected_node_names), (
        "Should return all available nodes except ALLOCATED nodes"
    )
    allocated_node_name = grouped_nodes[SlurmNodeState.ALLOCATED][0].name
    assert allocated_node_name not in returned_node_names, "ALLOCATED node should not be included"


def test_allocate_nodes_num_nodes_integers(
    slurm_system: SlurmSystem, grouped_nodes: dict[SlurmNodeState, list[SlurmNode]]
):
    group_name = "group_name"

    available_nodes = slurm_system.allocate_nodes(grouped_nodes, 1, group_name)
    expected_node_names = [grouped_nodes[SlurmNodeState.IDLE][0].name]

    returned_node_names = [node.name for node in available_nodes]

    assert len(returned_node_names) == len(expected_node_names), "Should return 1 available node"


def test_allocate_nodes_exceeding_limit(
    slurm_system: SlurmSystem, grouped_nodes: dict[SlurmNodeState, list[SlurmNode]]
):
    group_name = "group_name"
    num_nodes = 5
    available_nodes = 4

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"CloudAI is requesting {num_nodes} nodes from the group '{group_name}', but only "
            f"{available_nodes} nodes are available. Please review the available nodes in the system "
            f"and ensure there are enough resources to meet the requested node count. Additionally, "
            f"verify that the system can accommodate the number of nodes required by the test scenario."
        ),
    ):
        slurm_system.allocate_nodes(grouped_nodes, num_nodes, group_name)


@pytest.mark.parametrize(
    "stdout,stderr,is_completed",
    [
        ("COMPLETED", "", True),
        ("FAILED", "", True),
        ("CANCELLED", "", True),
        ("TIMEOUT", "", True),
        ("RUNNING", "", False),
        ("PENDING", "", False),
        ("COMPLETED RUNNING", "", False),
        ("RUNNING COMPLETED", "", False),
        ("COMPLETED COMPLETED", "", True),
        ("", "error", False),
    ],
)
def test_is_job_completed(stdout: str, stderr: str, is_completed: bool, slurm_system: SlurmSystem):
    job = BaseJob(test_run=Mock(), id=1)
    pp = Mock()
    pp.communicate = Mock(return_value=(stdout, stderr))
    slurm_system.cmd_shell.execute = Mock(return_value=pp)

    if stderr:
        with pytest.raises(RuntimeError):
            slurm_system.is_job_completed(job)
    else:
        assert slurm_system.is_job_completed(job) is is_completed


@pytest.mark.parametrize(
    "stdout,stderr,is_running",
    [
        ("RUNNING", "", True),
        ("PENDING", "", False),
        ("COMPLETED", "", False),
        ("FAILED", "", False),
        ("CANCELLED", "", False),
        ("TIMEOUT", "", False),
        ("", "error", False),
        ("   RUNNING \n   RUNNING \n   RUNNING \n COMPLETED \n    FAILED \n   RUNNING \n", "", True),
    ],
)
def test_is_job_running(stdout: str, stderr: str, is_running: bool, slurm_system: SlurmSystem):
    job = BaseJob(test_run=Mock(), id=1)
    pp = Mock()
    pp.communicate = Mock(return_value=(stdout, stderr))
    slurm_system.cmd_shell.execute = Mock(return_value=pp)

    if stderr:
        with pytest.raises(RuntimeError):
            slurm_system.is_job_running(job)
    else:
        assert slurm_system.is_job_running(job) is is_running


def test_is_job_running_with_retries(slurm_system: SlurmSystem):
    job = BaseJob(test_run=Mock(), id=1)
    command = f"sacct -j {job.id} --format=State --noheader"

    pp = Mock()
    pp.communicate = Mock(side_effect=[("", "Socket timed out"), ("", "slurm_load_jobs error"), ("RUNNING", "")])
    slurm_system.cmd_shell.execute = Mock(return_value=pp)

    assert slurm_system.is_job_running(job, retry_threshold=3) is True
    assert slurm_system.cmd_shell.execute.call_count == 3
    slurm_system.cmd_shell.execute.assert_called_with(command)


def test_is_job_running_exceeds_retries(slurm_system: SlurmSystem):
    job = BaseJob(test_run=Mock(), id=1)
    command = f"sacct -j {job.id} --format=State --noheader"

    # test known error in srderr
    pp = Mock()
    pp.communicate = Mock(return_value=("", "Socket timed out"))
    slurm_system.cmd_shell.execute = Mock(return_value=pp)
    with pytest.raises(RuntimeError):
        slurm_system.is_job_running(job)
    assert slurm_system.cmd_shell.execute.call_count == 3
    slurm_system.cmd_shell.execute.assert_called_with(command)

    # test unknown error in stderr
    pp.communicate = Mock(return_value=("", "FAILED"))
    slurm_system.cmd_shell.execute = Mock(return_value=pp)
    with pytest.raises(RuntimeError):
        slurm_system.is_job_running(job, retry_threshold=3)
    assert slurm_system.cmd_shell.execute.call_count == 1


def test_model_dump(slurm_system: SlurmSystem):
    sys_dict = slurm_system.model_dump()
    assert type(sys_dict["install_path"]) is str
    assert sys_dict["install_path"] == str(slurm_system.install_path)
    assert type(sys_dict["output_path"]) is str
    assert sys_dict["output_path"] == str(slurm_system.output_path)
    assert "cmd_shell" not in sys_dict
    recreated = SlurmSystem.model_validate(sys_dict)
    assert recreated.model_dump() == sys_dict


def test_default_partition_is_required():
    with pytest.raises(ValueError):
        SlurmSystem(name="", install_path=Path.cwd(), output_path=Path.cwd(), partitions=[])  # type: ignore


class TestParseNodes:
    def test_single_node(self, slurm_system: SlurmSystem):
        nodes = slurm_system.parse_nodes(["node01"])
        assert nodes == ["node01"]

    def test_two_nodes(self, slurm_system: SlurmSystem):
        nodes = slurm_system.parse_nodes(["node01", "node02"])
        assert nodes == ["node01", "node02"]

    def test_range(self, slurm_system: SlurmSystem):
        nodes = slurm_system.parse_nodes(["node0[1-3]"])
        assert nodes == ["node01", "node02", "node03"]

    def test_with_commas(self, slurm_system: SlurmSystem):
        nodes = slurm_system.parse_nodes(["node01,node02,node03"])
        assert nodes == ["node01", "node02", "node03"]

    @pytest.mark.parametrize("spec", ["part:", "part:group", "unknown:grp:1", "main:unknown:1"])
    def test_colon_invalid_syntax(self, slurm_system: SlurmSystem, spec: str):
        with pytest.raises(ValueError):
            slurm_system.parse_nodes([spec])


class TestGetNodesBySpec:
    def test_empty_nodes_list(self, slurm_system: SlurmSystem):
        num_nodes, node_list = slurm_system.get_nodes_by_spec(3, [])
        assert num_nodes == 3
        assert node_list == []

    @pytest.mark.parametrize(
        "in_nnodes,in_nodes,exp_nnodes,exp_nodes",
        [
            (2, ["node0[1-3]"], 3, ["node01", "node02", "node03"]),
            (4, ["node01,node02"], 2, ["node01", "node02"]),
            (1, ["node01,node02"], 2, ["node01", "node02"]),
        ],
    )
    @patch("cloudai.systems.slurm.slurm_system.SlurmSystem.parse_nodes")
    def test_explicit_node_names(
        self,
        mock_parse_nodes: Mock,
        slurm_system: SlurmSystem,
        in_nnodes: int,
        in_nodes: list[str],
        exp_nnodes: int,
        exp_nodes: list[str],
    ):
        mock_parse_nodes.return_value = exp_nodes

        num_nodes, node_list = slurm_system.get_nodes_by_spec(in_nnodes, in_nodes)

        mock_parse_nodes.assert_called_once_with(in_nodes)
        assert num_nodes == exp_nnodes
        assert node_list == exp_nodes


class ConcreteSlurmStrategy(SlurmCommandGenStrategy):
    def _container_mounts(self, tr: TestRun) -> list[str]:
        return []

    def generate_test_command(self, env_vars, cmd_args, tr):
        return ["test_command"]

    def job_name(self, tr: TestRun) -> str:
        return "job_name"


@pytest.fixture
def test_run(slurm_system: SlurmSystem) -> TestRun:
    test_run = TestRun(
        name="test_run",
        test=Test(
            test_definition=NCCLTestDefinition(
                name="test_run", description="test_run", test_template_name="nccl", cmd_args=NCCLCmdArgs()
            ),
            test_template=TestTemplate(slurm_system),
        ),
        num_nodes=2,
        nodes=["main:group1:2"],
        output_path=slurm_system.output_path,
    )

    test_run.output_path.mkdir(parents=True, exist_ok=True)

    return test_run


class TestSlurmCommandGenStrategyCache:
    @patch("cloudai.systems.slurm.SlurmSystem.get_nodes_by_spec")
    def test_strategy_caching(self, mock_get_nodes: Mock, slurm_system: SlurmSystem, test_run: TestRun):
        mock_get_nodes.return_value = (2, ["node01", "node02"])

        strategy = ConcreteSlurmStrategy(slurm_system, {})

        # First call to get nodes
        res = strategy.get_cached_nodes_spec(test_run)
        assert mock_get_nodes.call_count == 1
        assert res == (2, ["node01", "node02"])

        # Second call with same parameters should use cache
        res = strategy.get_cached_nodes_spec(test_run)
        assert mock_get_nodes.call_count == 1
        assert res == (2, ["node01", "node02"])

        # Different node spec should call get_nodes_by_spec again
        test_run.num_nodes = 1
        test_run.nodes = []
        strategy.get_cached_nodes_spec(test_run)
        assert mock_get_nodes.call_count == 2

        test_run.num_nodes = 2
        test_run.nodes = ["node01", "node03"]
        strategy.get_cached_nodes_spec(test_run)
        assert mock_get_nodes.call_count == 3

    @patch("cloudai.systems.slurm.SlurmSystem.get_nodes_by_spec")
    def test_per_test_isolation(self, mock_get_nodes: Mock, slurm_system: SlurmSystem, test_run: TestRun):
        mock_get_nodes.side_effect = [(2, ["node01", "node02"]), (2, ["node03", "node04"])]

        # Simulate two different test cases
        strategy1, strategy2 = ConcreteSlurmStrategy(slurm_system, {}), ConcreteSlurmStrategy(slurm_system, {})

        res = strategy1.get_cached_nodes_spec(test_run)
        assert mock_get_nodes.call_count == 1
        assert res == (2, ["node01", "node02"])

        res = strategy2.get_cached_nodes_spec(test_run)
        assert mock_get_nodes.call_count == 2
        assert res == (2, ["node03", "node04"])

        assert strategy1._node_spec_cache != strategy2._node_spec_cache, "Caches should be different"

    @patch("cloudai.systems.slurm.SlurmSystem.get_nodes_by_spec")
    def test_per_iteration_isolation(self, mock_get_nodes: Mock, slurm_system: SlurmSystem, test_run: TestRun):
        mock_get_nodes.side_effect = [(2, ["node01", "node02"]), (2, ["node03", "node04"])]

        strategy = ConcreteSlurmStrategy(slurm_system, {})

        res = strategy.get_cached_nodes_spec(test_run)
        assert mock_get_nodes.call_count == 1
        assert res == (2, ["node01", "node02"])

        test_run.current_iteration = 1
        res = strategy.get_cached_nodes_spec(test_run)
        assert mock_get_nodes.call_count == 2
        assert res == (2, ["node03", "node04"])

    @patch("cloudai.systems.slurm.SlurmSystem.get_nodes_by_spec")
    def test_per_step_isolation(self, mock_get_nodes: Mock, slurm_system: SlurmSystem, test_run: TestRun):
        mock_get_nodes.side_effect = [(2, ["node01", "node02"]), (2, ["node03", "node04"])]

        strategy = ConcreteSlurmStrategy(slurm_system, {})

        res = strategy.get_cached_nodes_spec(test_run)
        assert mock_get_nodes.call_count == 1
        assert res == (2, ["node01", "node02"])

        test_run.step = 1
        res = strategy.get_cached_nodes_spec(test_run)
        assert mock_get_nodes.call_count == 2
        assert res == (2, ["node03", "node04"])


@pytest.mark.parametrize(
    "scontrol_output,expected_support",
    [
        # Case 1: GresTypes includes gpu - should be supported
        (
            """Configuration data as of 2023-06-14T16:28:09
GresTypes               = gpu""",
            True,
        ),
        # Case 2: GresTypes is (null) - should NOT be supported
        (
            """Configuration data as of 2023-06-14T16:28:09
GresTypes               = (null)""",
            False,
        ),
        # Case 3: GresTypes does not include gpu - should NOT be supported
        (
            """Configuration data as of 2023-06-14T16:28:09
GresTypes               = cpu""",
            False,
        ),
        # Case 4: GresTypes includes multiple types including gpu - should be supported
        (
            """Configuration data as of 2023-06-14T16:28:09
GresTypes               = cpu,gpu,fpga""",
            True,
        ),
    ],
)
@patch("cloudai.systems.slurm.slurm_system.SlurmSystem.fetch_command_output")
def test_supports_gpu_directives(
    mock_fetch_command_output, scontrol_output: str, expected_support: bool, slurm_system: SlurmSystem
):
    mock_fetch_command_output.return_value = (scontrol_output, "")
    assert slurm_system.supports_gpu_directives == expected_support


@pytest.mark.parametrize(
    "cache_value",
    [True, False],
)
@patch("cloudai.systems.slurm.slurm_system.SlurmSystem.fetch_command_output")
def test_supports_gpu_directives_cache(mock_fetch_command_output, cache_value: bool, slurm_system: SlurmSystem):
    slurm_system.supports_gpu_directives_cache = cache_value
    assert slurm_system.supports_gpu_directives is cache_value
    mock_fetch_command_output.assert_not_called()


@pytest.mark.parametrize("enable,expected", [("true", True), ("false", False)])
def test_reports_spec_is_parsed(enable: str, expected: bool):
    spec = f"""
name = "example-cluster"
scheduler = "slurm"
install_path = "i"
output_path = "r"
default_partition = "p"
[[partitions]]
name = "p"

[reports]
per_test = {{ enable = {enable} }}
"""
    slurm = SlurmSystem.model_validate(toml.loads(spec))
    assert slurm.reports is not None
    assert isinstance(slurm.reports["per_test"], ReportConfig)
    assert slurm.reports["per_test"].enable is expected


@pytest.mark.parametrize("stdout,stderr, expected", [("", "error", None)])
def test_get_job_status(slurm_system: SlurmSystem, stdout: str, stderr: str, expected: tuple):
    job = BaseJob(test_run=Mock(), id=1)
    pp = Mock()
    pp.communicate = Mock(return_value=(stdout, stderr))
    slurm_system.cmd_shell.execute = Mock(return_value=pp)

    if stderr:
        with pytest.raises(RuntimeError):
            slurm_system.get_job_status(job)
    else:
        assert slurm_system.get_job_status(job) == expected


sacct_output = """2623913,job,COMPLETED,0:0,2025-05-09T01:34:52,2025-05-09T01:59:27,1475,sbatch sbatch_script.sh,
2623913.batch,batch,COMPLETED,0:0,2025-05-09T01:34:52,2025-05-09T01:59:27,1475,,
2623913.extern,extern,COMPLETED,0:0,2025-05-09T01:34:52,2025-05-09T01:59:27,1475,,
2623913.0,bash,COMPLETED,0:0,2025-05-09T01:35:24,2025-05-09T01:35:58,34,srun --export=ALL --mpi=pmix ...,
2623913.1,bash,COMPLETED,0:0,2025-05-09T01:35:58,2025-05-09T01:36:16,18,srun --export=ALL --mpi=pmix ...,
2623913.2,all_reduce_perf_mpi,COMPLETED,0:0,2025-05-09T01:36:16,2025-05-09T01:37:02,46,srun -N2 ...,
2623913.3,all_reduce_perf_mpi,COMPLETED,0:0,2025-05-09T01:37:02,2025-05-09T01:37:59,57,srun -N2 ...,
"""
sacct_output2 = """2968718|job:run|COMPLETED|0:0|2025-06-16T07:40:16|2025-06-16T07:49:08|532|sbatch run_submission.sh|
2968718.batch|batch|COMPLETED|0:0|2025-06-16T07:40:16|2025-06-16T07:49:08|532||
2968718.extern|extern|COMPLETED|0:0|2025-06-16T07:40:16|2025-06-16T07:49:08|532||
2968718.0|bash|COMPLETED|0:0|2025-06-16T07:40:54|2025-06-16T07:49:11|497|srun long cmd
with
  multiple
  lines |
"""


@pytest.mark.parametrize("sacct_output,delimiter,expected_nsteps", [(sacct_output, ",", 7), (sacct_output2, "|", 4)])
def test_slurm_job_metadata_from_sacct_output(sacct_output: str, delimiter: str, expected_nsteps: int):
    job_metadata = SlurmStepMetadata.from_sacct_output(sacct_output, delimiter=delimiter)
    assert len(job_metadata) == expected_nsteps


@pytest.mark.parametrize(
    "sinfo_nodes,squeue_nodes,expected_nodes",
    [
        ([], [], []),
        # sinfo nodes are always added
        (
            [SlurmNode(name="node01", partition="main", state=SlurmNodeState.IDLE)],
            [],
            [SlurmNode(name="node01", partition="main", state=SlurmNodeState.IDLE)],
        ),
        # squeue nodes only update the state
        ([], [SlurmNode(name="node01", partition="main", state=SlurmNodeState.ALLOCATED)], []),
        (
            [SlurmNode(name="node01", partition="main", state=SlurmNodeState.ALLOCATED)],
            [SlurmNode(name="node01", partition="main", state=SlurmNodeState.DRAINED)],
            [SlurmNode(name="node01", partition="main", state=SlurmNodeState.DRAINED)],
        ),
    ],
)
def test_update(
    slurm_system: SlurmSystem,
    sinfo_nodes: list[SlurmNode],
    squeue_nodes: list[SlurmNode],
    expected_nodes: list[SlurmNode],
):
    for partition in slurm_system.partitions:
        partition.slurm_nodes.clear()

    mod_path = "cloudai.systems.slurm.slurm_system.SlurmSystem"
    with (
        patch(f"{mod_path}.nodes_from_sinfo", return_value=sinfo_nodes) as mock_nodes_from_sinfo,
        patch(f"{mod_path}.nodes_from_squeue", return_value=squeue_nodes) as mock_nodes_from_squeue,
    ):
        slurm_system.update()
        mock_nodes_from_sinfo.assert_called_once()
        mock_nodes_from_squeue.assert_called_once()

        all_nodes_set = set([node for p in slurm_system.partitions for node in p.slurm_nodes])
        assert all_nodes_set == set(expected_nodes)
