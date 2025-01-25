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
from typing import Union

import pytest
import toml

from cloudai import Parser, Registry
from cloudai.test_definitions import ChakraReplayCmdArgs, NCCLCmdArgs, NCCLTestDefinition
from cloudai.test_definitions.chakra_replay import ChakraReplayTestDefinition
from cloudai.test_definitions.gpt import GPTCmdArgs, GPTFdl, GPTTestDefinition
from cloudai.test_definitions.grok import GrokCmdArgs, GrokFdl, GrokTestDefinition
from cloudai.test_definitions.nemo_launcher import NeMoLauncherCmdArgs, NeMoLauncherTestDefinition
from cloudai.test_definitions.nemotron import NemotronCmdArgs, NemotronFdl, NemotronTestDefinition
from cloudai.test_definitions.ucc import UCCCmdArgs, UCCTestDefinition
from tests.conftest import MyTestDefinition

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
    t = MyTestDefinition(name="test", description="test", test_template_name="test", cmd_args={}, extra_cmd_args=input)
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

    Parser.parse_tests([toml_file], None)  # type: ignore


def test_chakra_docker_image_is_required():
    with pytest.raises(ValueError) as exc_info:
        ChakraReplayCmdArgs.model_validate({})
    assert "Field required" in str(exc_info.value)
    assert "docker_image_url" in str(exc_info.value)


@pytest.mark.parametrize(
    "test",
    [
        UCCTestDefinition(name="ucc", description="desc", test_template_name="ucc", cmd_args=UCCCmdArgs()),
        NCCLTestDefinition(name="nccl", description="desc", test_template_name="nccl", cmd_args=NCCLCmdArgs()),
        GPTTestDefinition(
            name="gpt",
            description="desc",
            test_template_name="gpt",
            cmd_args=GPTCmdArgs(
                fdl_config="",
                docker_image_url="fake://url/gpt",
                fdl=GPTFdl(ici_mesh_shape="[1, 1, 1, 1]", dcn_mesh_shape="[1, 1, 1, 1]"),
            ),
        ),
        GrokTestDefinition(
            name="grok",
            description="desc",
            test_template_name="grok",
            cmd_args=GrokCmdArgs(
                docker_image_url="fake://url/grok",
                fdl_config="",
                fdl=GrokFdl(ici_mesh_shape="[1, 1, 1, 1]", dcn_mesh_shape="[1, 1, 1, 1]"),
            ),
        ),
        NemotronTestDefinition(
            name="nemotron",
            description="desc",
            test_template_name="nemotron",
            cmd_args=NemotronCmdArgs(
                docker_image_url="fake://url/nemotron",
                fdl_config="",
                fdl=NemotronFdl(ici_mesh_shape="[1, 1, 1, 1]", dcn_mesh_shape="[1, 1, 1, 1]"),
            ),
        ),
        NeMoLauncherTestDefinition(
            name="nemo", description="desc", test_template_name="nemo", cmd_args=NeMoLauncherCmdArgs()
        ),
        ChakraReplayTestDefinition(
            name="chakra",
            description="desc",
            test_template_name="chakra",
            cmd_args=ChakraReplayCmdArgs(docker_image_url="fake://url/chakra"),
        ),
    ],
)
def test_docker_installable_persists(
    test: Union[
        ChakraReplayTestDefinition,
        GPTTestDefinition,
        GrokTestDefinition,
        NCCLTestDefinition,
        NeMoLauncherTestDefinition,
        NemotronTestDefinition,
        UCCTestDefinition,
    ],
    tmp_path: Path,
):
    test.docker_image.installed_path = tmp_path
    assert test.docker_image.installed_path == tmp_path


@pytest.mark.parametrize(
    "test",
    [
        NeMoLauncherTestDefinition(
            name="nemo", description="desc", test_template_name="nemo", cmd_args=NeMoLauncherCmdArgs()
        )
    ],
)
def test_python_executable_installable_persists(test: NeMoLauncherTestDefinition, tmp_path: Path):
    test.python_executable.git_repo.installed_path = tmp_path
    test.python_executable.venv_path = tmp_path
    assert test.python_executable.git_repo.installed_path == tmp_path
    assert test.python_executable.venv_path == tmp_path
