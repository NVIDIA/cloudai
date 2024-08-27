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
from cloudai import Registry, TestDefinition, TestParser

TOML_FILES = list(Path("conf").glob("**/*.toml"))
ALL_TESTS = [t for t in TOML_FILES if "test_template_name" in t.read_text()]


def test_extra_args_str():
    t = TestDefinition(
        name="test", description="test", test_template_name="test", cmd_args={}, extra_cmd_args={"-a": "1"}
    )
    assert t.extra_args_str() == "-a=1"


@pytest.mark.parametrize("toml_file", ALL_TESTS, ids=lambda x: str(x))
def test_all_tests(toml_file: Path):
    with toml_file.open("r") as f:
        toml_dict = toml.load(f)

    registry = Registry()
    template_name = toml_dict["test_template_name"]
    assert template_name in registry.test_definitions_map, f"Unknown test template: {template_name}"

    tp = TestParser(Path(), {})
    tp.load_test_definition(toml.load(toml_file))
