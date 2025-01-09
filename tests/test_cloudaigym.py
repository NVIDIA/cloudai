import unittest
from unittest.mock import MagicMock

from gymnasium import spaces

from cloudai._core.test_scenario import TestRun, TestScenario
from cloudai.environment.cloudai_gym import CloudAIGymEnv
from cloudai.systems import SlurmSystem


class TestCloudAIGymEnv(unittest.TestCase):
    def setUp(self):
        # Mock the TestRun object
        self.test_run = MagicMock(spec=TestRun)
        self.test_run.test.cmd_args = {
            "docker_image_url": "https://docker/url",
            "load_container": True,
            "output_path": "results/dse_jaxtoolbox_grok_2025-01-09_04-58-39/Tests.1/0",
            "Grok.fdl": {
                "checkpoint_policy": ['\\"save_iteration_input\\"', '\\"save_none\\"'],
                "num_gpus": [1, 8, 16],
            },
        }

        # Mock the SlurmSystem and TestScenario objects
        self.system = MagicMock(spec=SlurmSystem)
        self.test_scenario = MagicMock(spec=TestScenario)

        # Initialize CloudAIGymEnv
        self.env = CloudAIGymEnv(test_run=self.test_run, system=self.system, test_scenario=self.test_scenario)

    def test_action_space(self):
        # Expected action space
        expected_action_space = spaces.Dict(
            {"Grok.fdl.checkpoint_policy": spaces.Discrete(2), "Grok.fdl.num_gpus": spaces.Discrete(3)}
        )

        # Assert that the action space is set correctly
        self.assertEqual(self.env.action_space.spaces.keys(), expected_action_space.spaces.keys())
        for key in expected_action_space.spaces.keys():
            self.assertEqual(type(self.env.action_space.spaces[key]), type(expected_action_space.spaces[key]))
            self.assertEqual(self.env.action_space.spaces[key].n, expected_action_space.spaces[key].n)


if __name__ == "__main__":
    unittest.main()
