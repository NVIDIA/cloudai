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
from cloudai.test_definitions import ChakraReplayCmdArgs, NCCLCmdArgs, NCCLTestDefinition
from cloudai.test_definitions.gpt import GPTCmdArgs, GPTTestDefinition
from cloudai.test_definitions.grok import (
    GrokCmdArgs,
    GrokTestDefinition,
)
from cloudai.test_definitions.jax_toolbox import JaxFdl

TOML_FILES = list(Path("conf").glob("**/*.toml"))
ALL_TESTS = [t for t in TOML_FILES if "test_template_name" in t.read_text()]


@pytest.mark.parametrize(
    "input,expected",
    [
        ({"-a": "1"}, "-a=1"),
        ({"-a": ""}, "-a"),
        ({"-a": "1", "-b": "2"}, "-a=1 -b=2"),
        ({"-a": "1", "-b": "2", "-c": ""}, "-a=1 -b=2 -c"),
    ],
)
def test_extra_args_str(input: dict, expected: str):
    t = TestDefinition(name="test", description="test", test_template_name="test", cmd_args={}, extra_cmd_args=input)
    assert t.extra_args_str == expected


@pytest.mark.parametrize(
    "input,expected",
    [
        ({"-a": "1"}, "-a 1"),
        ({"-a": ""}, "-a"),
        ({"-a": "1", "-b": "2"}, "-a 1 -b 2"),
        ({"-a": "1", "-b": "2", "-c": ""}, "-a 1 -b 2 -c"),
    ],
)
def test_extra_args_str_nccl(input: dict, expected: str):
    t = NCCLTestDefinition(
        name="test", description="test", test_template_name="test", cmd_args=NCCLCmdArgs(), extra_cmd_args=input
    )
    assert t.extra_args_str == expected


@pytest.mark.parametrize("toml_file", ALL_TESTS, ids=lambda x: str(x))
def test_all_tests(toml_file: Path):
    with toml_file.open("r") as f:
        toml_dict = toml.load(f)

    registry = Registry()
    template_name = toml_dict["test_template_name"]
    assert template_name in registry.test_definitions_map, f"Unknown test template: {template_name}"

    TestParser.load_test_definition(toml.load(toml_file))


def test_chakra_docker_image_is_required():
    with pytest.raises(ValueError) as exc_info:
        ChakraReplayCmdArgs.model_validate({})
    assert "Field required" in str(exc_info.value)
    assert "docker_image_url" in str(exc_info.value)


def test_gpt_test_definition_cmd_args_dict():
    gpt = GPTTestDefinition(
        name="gpt",
        description="gpt",
        test_template_name="gpt",
        cmd_args=GPTCmdArgs(fdl_config=""),
    )

    cargs = gpt.cmd_args_dict

    assert "GPT.fdl" in cargs
    assert "GPT.setup_flags" in cargs
    assert "GPT.XLA_FLAGS" in cargs

    assert "pre_test" in cargs
    assert "GPT.pre_test" not in cargs


def test_grok_test_definition_cmd_args_dict():
    grok = GrokTestDefinition(
        name="grok",
        description="grok",
        test_template_name="grok",
        cmd_args=GrokCmdArgs(),
    )

    cargs = grok.cmd_args_dict

    assert "Grok.setup_flags" in cargs
    assert "Grok.enable_pgle" in cargs
    assert "Grok.fdl" in cargs

    assert "pre_test" in cargs
    assert "Grok.pre_test" not in cargs

    assert "Grok.profile" in cargs
    assert "XLA_FLAGS" in cargs["Grok.profile"]
    assert "Grok.perf" in cargs
    assert "XLA_FLAGS" in cargs["Grok.perf"]


def test_jax_fprop_dtype_is_escaped():
    fdl = JaxFdl()
    d = fdl.model_dump()
    assert d["fprop_dtype"] == f'\\"{fdl.fprop_dtype}\\"'
