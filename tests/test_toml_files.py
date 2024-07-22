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

import pytest
import toml
from cloudai._core.registry import Registry

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


ALL_SYSTEMS = list(Path("conf/system").glob("*.toml"))


@pytest.mark.parametrize("system_file", ALL_SYSTEMS, ids=lambda x: str(x))
def test_systems(system_file: Path):
    """
    Validate the syntax of a system configuration file.

    Args:
        system_file (Path): The path to the system configuration file to validate.
    """
    registry = Registry()

    with system_file.open("r") as f:
        data = toml.load(f)

    scheduler = data.get("scheduler", "").lower()
    if scheduler not in registry.systems_map:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    system = registry.systems_map[scheduler](**data)
    assert system is not None
