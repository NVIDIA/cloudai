import numpy as np
import pytest

from src.cloudai.environment.example_environment import ExampleEnv


# Fixture to initialize the environment
@pytest.fixture
def env():
    """
    Fixture to provide a fresh instance of ParameterConvergeEnv.
    Automatically resets the environment before each test.
    """
    environment = ExampleEnv(max_steps=10)
    environment.reset()
    return environment


def test_environment_initialization(env):
    """
    Test that the environment initializes correctly.
    """
    observation, info = env.reset()
    assert isinstance(observation, np.ndarray), "Observation should be a numpy array."
    assert observation.shape == (3,), "Observation should have three elements."
    assert env.current_step == 0, "Step count should initialize to 0."
    assert not env.done, "Environment should not be done immediately after reset."


@pytest.mark.parametrize(
    "action, expected_done",
    [
        ({"num_cores": 4, "freq": 2.0, "mem_type": 1, "mem_size": 32}, False),
        ({"num_cores": 10, "freq": 1.5, "mem_type": 0, "mem_size": 16}, False),
    ],
)
def test_step_execution(env, action, expected_done):
    """
    Test that a valid action updates the environment correctly.
    """
    observation, reward, done, truncated, info = env.step(action)
    assert isinstance(observation, np.ndarray), "Observation should be a numpy array."
    assert isinstance(reward, float), "Reward should be a float."
    assert reward < 0, "Reward should be negative for L2-norm distance."
    assert done == expected_done, "Done flag should match expected behavior."
    assert not truncated, "Truncated should always be False."


@pytest.mark.parametrize(
    "action",
    [
        {"num_cores": 20, "freq": 5.0, "mem_type": 5, "mem_size": 100},  # All invalid
        {"num_cores": 15, "freq": 0.0, "mem_type": -1, "mem_size": 65},  # Boundary invalid
    ],
)
def test_invalid_action(env, action):
    """
    Test that invalid actions raise a ValueError.
    """
    with pytest.raises(ValueError):
        env.step(action)


def test_episode_completion(env):
    """
    Test that the environment ends the episode after max_steps.
    """
    action = {"num_cores": 4, "freq": 2.0, "mem_type": 1, "mem_size": 32}
    done = False
    for _ in range(10):  # Perform max_steps
        observation, reward, done, truncated, info = env.step(action)
    assert done, "Environment should be done after reaching max_steps."
    observation, info = env.reset()
    assert env.current_step == 0, "Step count should reset to 0 after reset."
    assert not env.done, "Environment should not be done after reset."


def test_reset_behavior(env):
    """
    Test that resetting the environment restores it to the initial state.
    """
    observation, info = env.reset()
    assert np.allclose(observation, [0, 0, 0]), "Initial observation should be zeros."
    assert env.current_step == 0, "Step count should be reset to 0."
