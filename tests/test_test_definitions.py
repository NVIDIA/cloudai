# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Union, cast

import pytest
import toml
from pydantic import ValidationError

from cloudai import NsysConfiguration, Parser, Registry, TestConfigParsingError, TestParser
from cloudai.workloads.chakra_replay import ChakraReplayCmdArgs, ChakraReplayTestDefinition
from cloudai.workloads.jax_toolbox import (
    GPTCmdArgs,
    GPTTestDefinition,
    GrokCmdArgs,
    GrokTestDefinition,
    NemotronCmdArgs,
    NemotronTestDefinition,
)
from cloudai.workloads.megatron_run import MegatronRunCmdArgs, MegatronRunTestDefinition
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition
from cloudai.workloads.nemo_launcher import NeMoLauncherCmdArgs, NeMoLauncherTestDefinition
from cloudai.workloads.nemo_run import NeMoBaseRunCmdArgs, NeMoRunTestDefinition
from cloudai.workloads.ucc_test import UCCCmdArgs, UCCTestDefinition
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
            cmd_args=GPTCmdArgs(fdl_config="", docker_image_url="fake://url/gpt"),
        ),
        GrokTestDefinition(
            name="grok",
            description="desc",
            test_template_name="grok",
            cmd_args=GrokCmdArgs(docker_image_url="fake://url/grok"),
        ),
        NemotronTestDefinition(
            name="nemotron",
            description="desc",
            test_template_name="nemotron",
            cmd_args=NemotronCmdArgs(docker_image_url="fake://url/nemotron"),
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


class TestNsysConfiguration:
    def test_default(self):
        nsys = NsysConfiguration()
        assert nsys.enable is True
        assert nsys.nsys_binary == "nsys"
        assert nsys.task == "profile"

    def test_cmd_args(self):
        nsys = NsysConfiguration()
        assert nsys.cmd_args == ["nsys", "profile"]

    @pytest.mark.parametrize("value", [True, False])
    def test_force_overwrite(self, value: bool):
        nsys = NsysConfiguration(force_overwrite=value)
        assert nsys.cmd_args == ["nsys", "profile", f"--force-overwrite={'true' if value else 'false'}"]

    def test_extra_args(self):
        nsys = NsysConfiguration(extra_args=["--extra", "args"])
        assert nsys.cmd_args == ["nsys", "profile", "--extra", "args"]


class TestLoadTestDefinition:
    @pytest.fixture
    def test_parser(self) -> TestParser:
        tp = TestParser([], None)  # type: ignore
        tp.current_file = Path(__file__)
        return tp

    @pytest.fixture
    def nemorun_with_unknown_field(self) -> dict:
        return {
            "name": "n",
            "description": "d",
            "test_template_name": "NeMoRun",
            "cmd_args": {
                **NeMoBaseRunCmdArgs(
                    docker_image_url="fake://url/nemo", task="task", recipe_name="recipe"
                ).model_dump(),
                "unknown": {"sub": "sub"},
                "trainer": {"strategy": {"nested_unknown": "nested_unknown"}},
            },
        }

    def test_load_test_definition(self, test_parser: TestParser, nemorun_with_unknown_field: dict):
        test_def: NeMoRunTestDefinition = cast(
            NeMoRunTestDefinition, test_parser.load_test_definition(data=nemorun_with_unknown_field)
        )
        assert test_def.docker_image.url == "fake://url/nemo"
        assert test_def.cmd_args.task == "task"
        assert test_def.cmd_args.recipe_name == "recipe"
        assert test_def.cmd_args.unknown["sub"] == "sub"  # type: ignore
        assert test_def.cmd_args.trainer.strategy.nested_unknown == "nested_unknown"  # type: ignore

    def test_load_test_definition_strict(self, test_parser: TestParser, nemorun_with_unknown_field: dict):
        with pytest.raises(TestConfigParsingError) as exc_info:
            test_parser.load_test_definition(data=nemorun_with_unknown_field, strict=True)
        assert "Failed to parse test spec using strict mode" in str(exc_info.value)

    def test_load_test_definition_unknown_test(self, test_parser: TestParser):
        with pytest.raises(NotImplementedError) as exc_info:
            test_parser.load_test_definition(data={"test_template_name": "unknown"})
        assert "TestTemplate with name 'unknown' not supported." in str(exc_info.value)


class TestMegatronRun:
    @pytest.fixture
    def megatron_run(self) -> MegatronRunTestDefinition:
        return MegatronRunTestDefinition(
            name="mr",
            description="desc",
            test_template_name="mr",
            cmd_args=MegatronRunCmdArgs(docker_image_url="fake://url/mr", run_script=Path(__file__)),
        )

    def test_default(self, megatron_run: MegatronRunTestDefinition):
        cmd = " ".join([f"{k} {v}" for k, v in megatron_run.cmd_args_dict.items()])
        assert "--global-batch-size 16" in cmd
        assert "--hidden-size 4096" in cmd
        assert "--max-position-embeddings 4096" in cmd
        assert "--num-attention-heads 32" in cmd
        assert "--num-layers 32" in cmd
        assert "--pipeline-model-parallel-size 1" in cmd
        assert "--recompute-activations " in cmd
        assert "--seq-length 4096" in cmd
        assert "--tensor-model-parallel-size 2" in cmd

    def test_nones_are_dropped(self, megatron_run: MegatronRunTestDefinition):
        to_be_none = {
            "hidden_size": None,
            "max_position_embeddings": None,
            "num_attention_heads": None,
            "num_layers": None,
            "pipeline_model_parallel_size": None,
            "recompute_activations": None,
            "seq_length": None,
            "tensor_model_parallel_size": None,
        }
        megatron_run.cmd_args = MegatronRunCmdArgs.model_validate(
            {"docker_image_url": "fake://url/mr", "run_script": "/path/to/script", **to_be_none}
        )

        cmd = " ".join([f"{k} {v}" for k, v in megatron_run.cmd_args_dict.items()])
        for arg in to_be_none:
            assert f"--{arg.replace('_', '-')}" not in cmd

    def test_unknowns_are_handled(self, megatron_run: MegatronRunTestDefinition):
        megatron_run.cmd_args = MegatronRunCmdArgs.model_validate(
            {
                "docker_image_url": "fake://url/mr",
                "run_script": "/path/to/script",
                "unknown": "1",
                "with_underscore": "2.5",
            }
        )

        cmd = " ".join([f"{k} {v}" for k, v in megatron_run.cmd_args_dict.items()])
        assert "--unknown 1" in cmd
        assert "--with-underscore 2.5" in cmd

    def test_dashed_args(self):
        with pytest.raises(ValidationError):
            MegatronRunCmdArgs.model_validate({"docker_image_url": "fake://url/mr", "with-dash": "value"})

    def test_tokenizer_model(self, megatron_run: MegatronRunTestDefinition):
        megatron_run.cmd_args = MegatronRunCmdArgs.model_validate(
            {
                "docker_image_url": "fake://url/mr",
                "run_script": "/path/to/script",
                "tokenizer_model": "/path/to/tokenizer",
            }
        )
        assert megatron_run.cmd_args_dict["--tokenizer-model"] == Path("/path/to/tokenizer")
        assert megatron_run.cmd_args.tokenizer_model == Path("/path/to/tokenizer")

    def test_auxiliary_fields_not_in_model_dump(self, megatron_run: MegatronRunTestDefinition):
        d = megatron_run.cmd_args.model_dump()
        assert "docker_image_url" not in d
        assert "run_script" not in d
