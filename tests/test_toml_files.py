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
from unittest.mock import patch

import pytest
import toml

from cloudai import Parser
from cloudai.models import TestScenarioModel

TOML_FILES = list(Path("conf").glob("**/*.toml"))


@pytest.mark.parametrize("toml_file", TOML_FILES, ids=lambda x: str(x))
def test_toml_files(toml_file: Path):
    """
    Validate the syntax of a .toml file.

    Args:
        toml_file (Path): The path to the .toml file to validate.
    """
    with toml_file.open("r") as f:
        assert toml.load(f) is not None


ALL_SYSTEMS = [p for p in Path("conf/").glob("**/*.toml") if "scheduler =" in p.read_text()]


@pytest.mark.parametrize("system_file", ALL_SYSTEMS, ids=lambda x: str(x))
@patch("kubernetes.config.load_kube_config")
@patch("pathlib.Path.exists", return_value=True)
def test_systems(mock_exists, mock_load_kube_config, system_file: Path):
    """
    Validate the syntax of a system configuration file.

    Args:
        system_file (Path): The path to the system configuration file to validate.
    """
    mock_load_kube_config.return_value = None
    system = Parser(system_file).parse_system(system_file)
    assert system is not None


ALL_TEST_SCENARIOS = [p for p in Path("conf/").glob("**/*.toml") if "[[Tests]]" in p.read_text()]


@pytest.mark.parametrize("test_scenario_file", ALL_TEST_SCENARIOS, ids=lambda x: str(x))
def test_test_scenarios(test_scenario_file: Path):
    """
    Validate the syntax of a test scenario file.

    Args:
        test_scenario_file (Path): The path to the test scenario file to validate.
    """
    with test_scenario_file.open("r") as f:
        d = toml.load(f)
        TestScenarioModel.model_validate(d)
