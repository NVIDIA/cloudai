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

from itertools import product
from pathlib import Path
from typing import Dict, List, Union

import pytest
from cloudai._core.base_job import BaseJob
from cloudai._core.system import System
from cloudai._core.test import Test, TestDefinition, TestTemplate
from pydantic import ValidationError


class MockSystem(System):
    """
    Mock subclass of System for testing purposes.
    """

    def update(self) -> None:
        pass

    def is_job_running(self, job: "BaseJob") -> bool:
        return False

    def is_job_completed(self, job: "BaseJob") -> bool:
        return True

    def kill(self, job: "BaseJob") -> None:
        pass


class MockTestTemplate(TestTemplate):
    """
    Mock subclass of TestTemplate for testing purposes.
    """

    pass


class ConcreteTestDefinition(TestDefinition):
    """
    Concrete subclass of TestDefinition for testing purposes.
    """

    @property
    def installables(self):
        return []


# Mock generate commands that will be handled within the agent later
def generate_commands(cmd_args: Dict[str, Union[str, List[str]]]) -> List[str]:
    # Handle empty cmd_args
    if not cmd_args:
        return []

    keys, values = zip(*[(key, value if isinstance(value, list) else [value]) for key, value in cmd_args.items()])

    commands = []
    for combination in product(*values):
        cmd_parts = [f"--{key}={val}" for key, val in zip(keys, combination)]
        commands.append(" ".join(cmd_parts))
    return commands


def test_cmd_args_with_ranges():
    """Test that cmd_args with ranges is correctly parsed and retains lists."""
    data = {
        "name": "example DSE",
        "description": "Example DSE",
        "test_template_name": "ExampleEnv",
        "cmd_args": {
            "a": [1, 16],
            "b": [1, 2, 4, 8],
            "c": [1, 2, 4, 8, 16],
            "d": [1, 2, 4, 8],
            "e": [10, 100, 500],
            "f": [10, 100, 500],
            "num_layers": "4",
            "use_fp8": "1",
        },
        "extra_env_vars": {
            "ENV1": "0",
            "ENV2": "1",
            "ENV3": "3221225472",
        },
    }

    # Create ConcreteTestDefinition
    test_def = ConcreteTestDefinition(**data)

    # Assert cmd_args retain ranges
    assert test_def.cmd_args_dict == {
        "a": [1, 16],
        "b": [1, 2, 4, 8],
        "c": [1, 2, 4, 8, 16],
        "d": [1, 2, 4, 8],
        "e": [10, 100, 500],
        "f": [10, 100, 500],
        "num_layers": "4",
        "use_fp8": "1",
    }

    mock_system = MockSystem(
        name="mock_system",
        scheduler="mock_scheduler",
        install_path=Path("/mock/install/path"),
        output_path=Path("/mock/output/path"),
    )
    # Create a Test instance
    test_template = MockTestTemplate(system=mock_system, name="example_template")
    test = Test(test_definition=test_def, test_template=test_template)

    # Assert cmd_args are the same as raw_cmd_args since we are retaining lists
    assert test.cmd_args == test_def.cmd_args_dict


def test_cmd_args_with_static_values():
    """Test that cmd_args with static values is correctly parsed."""
    data = {
        "name": "example DSE",
        "description": "Example DSE",
        "test_template_name": "ExampleEnv",
        "cmd_args": {
            "a": "1",
            "b": "4",
            "c": "8",
            "d": "2",
            "e": "10",
            "f": "500",
            "num_layers": "4",
            "use_fp8": "1",
        },
        "extra_env_vars": {
            "ENV1": "0",
            "ENV2": "1",
            "ENV3": "3221225472",
        },
    }

    # Create ConcreteTestDefinition
    test_def = ConcreteTestDefinition(**data)

    # Assert cmd_args retain static values
    assert test_def.cmd_args_dict == {
        "a": "1",
        "b": "4",
        "c": "8",
        "d": "2",
        "e": "10",
        "f": "500",
        "num_layers": "4",
        "use_fp8": "1",
    }

    mock_system = MockSystem(
        name="mock_system",
        scheduler="mock_scheduler",
        install_path=Path("/mock/install/path"),
        output_path=Path("/mock/output/path"),
    )
    # Create a Test instance
    test_template = MockTestTemplate(system=mock_system, name="example_template")
    test = Test(test_definition=test_def, test_template=test_template)

    # Assert cmd_args are the same as raw_cmd_args since they are static values
    assert test.cmd_args == test_def.cmd_args_dict


def test_generate_commands_with_ranges():
    """Test command generation when cmd_args contain ranges."""
    cmd_args = {
        "a": [1, 16],
        "b": [1, 2],
        "flag": "true",
        "layers": "4",
    }

    commands = generate_commands(cmd_args)

    # This is a subset of the expected commands for all combinations
    expected_commands = [
        "--a=1 --b=1 --flag=true --layers=4",
        "--a=1 --b=2 --flag=true --layers=4",
        "--a=16 --b=1 --flag=true --layers=4",
        "--a=16 --b=2 --flag=true --layers=4",
    ]

    assert commands == expected_commands


def test_generate_commands_with_static_values():
    """Test command generation when cmd_args contain only static values."""
    cmd_args = {
        "a": ["1"],
        "b": ["4"],
        "flag": "false",
        "layers": "2",
    }

    commands = generate_commands(cmd_args)

    # Expected single command
    expected_commands = [
        "--a=1 --b=4 --flag=false --layers=2",
    ]

    assert commands == expected_commands


def test_generate_commands_with_empty_cmd_args():
    """Test command generation with empty cmd_args."""
    cmd_args = {}

    commands = generate_commands(cmd_args)

    # No commands should be generated
    assert commands == []


def test_invalid_toml_parsing_missing_fields():
    """Test that TestParser raises an error for TOML data missing required fields."""
    # Invalid data: missing required 'name' field
    invalid_toml_data = {
        "description": "Example invalid TOML",
        "test_template_name": "ExampleEnv",
        "cmd_args": {
            "a": "1",
        },
    }

    # Simulate TestDefinition validation
    with pytest.raises(ValidationError, match="name\n  Field required"):
        ConcreteTestDefinition(**invalid_toml_data)


def test_invalid_toml_parsing_unexpected_field():
    """Test that TestDefinition raises an error for unexpected fields in TOML."""
    # Invalid data: includes an unexpected field
    invalid_toml_data = {
        "name": "example DSE",
        "description": "Example invalid TOML",
        "test_template_name": "ExampleEnv",
        "cmd_args": {
            "a": "1",
        },
        "unexpected_field": "unexpected_value",  # Unexpected field
    }

    # Validate that a ValidationError is raised for extra fields
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ConcreteTestDefinition(**invalid_toml_data)
