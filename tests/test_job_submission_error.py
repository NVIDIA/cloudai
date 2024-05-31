import subprocess
from unittest.mock import MagicMock, Mock

import pytest
from cloudai._core.exceptions import JobIdRetrievalError
from cloudai._core.test import Test
from cloudai._core.test_scenario import TestScenario
from cloudai._core.test_template import TestTemplate
from cloudai.runner.slurm.slurm_runner import SlurmRunner
from cloudai.systems import SlurmSystem
from cloudai.systems.slurm import SlurmNode, SlurmNodeState
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
        self.name = "Mock Test"
        self.description = "A mock test description"
        self.test_template = MagicMock(spec=TestTemplate)
        self.env_vars = {}
        self.cmd_args = {}
        self.extra_env_vars = {}
        self.extra_cmd_args = ""
        self.section_name = "Tests.1"
        self.current_iteration = 0

    def gen_exec_command(self, output_path):
        return "sbatch mock_script.sh"

    def get_job_id(self, stdout, stderr):
        return None


@pytest.fixture
def slurm_system(tmpdir):
    nodes = [
        SlurmNode(name="nodeA001", partition="main", state=SlurmNodeState.UNKNOWN_STATE),
        SlurmNode(name="nodeB001", partition="main", state=SlurmNodeState.UNKNOWN_STATE),
    ]
    system = SlurmSystem(
        name="test_system",
        install_path=tmpdir,
        output_path=tmpdir,
        default_partition="main",
        partitions={"main": nodes},
    )
    return system


@pytest.fixture
def slurm_runner(slurm_system):
    test_scenario = TestScenario(name="Test Scenario", tests=[MockTest(section_name="Mock Test")])
    runner = SlurmRunner(mode="run", system=slurm_system, test_scenario=test_scenario)
    runner.cmd_shell = MockCommandShell()
    return runner


def test_job_id_retrieval_error(slurm_runner):
    test = slurm_runner.test_scenario.tests[0]
    with pytest.raises(JobIdRetrievalError) as excinfo:
        slurm_runner._submit_test(test)
    assert "Failed to retrieve job ID from command output." in str(excinfo.value)
    assert "sbatch: error: Batch job submission failed: Requested node configuration is not available" in str(
        excinfo.value
    )
