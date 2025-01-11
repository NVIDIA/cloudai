from unittest.mock import MagicMock

import pytest

from cloudai._core.configurator.cloudai_gym import CloudAIGymEnv
from cloudai._core.test_scenario import TestRun, TestScenario
from cloudai.systems import SlurmSystem


@pytest.fixture
def setup_env():
    test_run = MagicMock(spec=TestRun)
    system = MagicMock(spec=SlurmSystem)
    test_scenario = MagicMock(spec=TestScenario)

    test_run.test = MagicMock()
    test_run.test.cmd_args = {
        "docker_image_url": "https://docker/url",
        "iters": [10, 100],
        "maxbytes": [1024, 2048],
        "minbytes": [512, 1024, 2048, 4096],
        "ngpus": [4],
        "subtest_name": "nccl_test",
        "warmup_iters": 5,
    }

    return test_run, system, test_scenario


def test_action_space_nccl(setup_env):
    test_run, system, test_scenario = setup_env
    env = CloudAIGymEnv(test_run=test_run, system=system, test_scenario=test_scenario)
    action_space = env.define_action_space()

    expected_action_space = {
        "iters": 2,
        "maxbytes": 2,
        "minbytes": 4,
        "ngpus": 1,
    }

    assert action_space.keys() == expected_action_space.keys()
    for key in expected_action_space:
        assert action_space[key] == expected_action_space[key]


def test_observation_space(setup_env):
    test_run, system, test_scenario = setup_env
    env = CloudAIGymEnv(test_run=test_run, system=system, test_scenario=test_scenario)
    observation_space = env.define_observation_space()

    expected_observation_space = [0.0]

    assert observation_space == expected_observation_space
