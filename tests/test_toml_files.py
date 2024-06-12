# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import json
from pathlib import Path

import jsonschema
import jsonschema.validators
import pytest
import toml

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


TEST_SCENARIO_FILES = list(Path("conf/v0.6/general/test_scenario").rglob("*.toml"))
TEST_SCENARIO_SCHEMA = json.loads(Path("schemas/test_scenario.schema.json").read_text())


@pytest.mark.parametrize("toml_file", TEST_SCENARIO_FILES, ids=lambda x: str(x))
def test_scenarios(toml_file: Path):
    """
    Validate test scenarios against the schema.

    Args:
        toml_file (Path): The path to the .toml file to validate.
    """
    with toml_file.open("r") as f:
        data = toml.load(f)

    jsonschema.validate(data, TEST_SCENARIO_SCHEMA)
