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
from unittest.mock import Mock

import pytest

from cloudai._core.test import Test
from cloudai._core.test_scenario import TestRun
from cloudai.schema.test_template.nemo_run.slurm_command_gen_strategy import NeMoRunSlurmCommandGenStrategy
from cloudai.systems import SlurmSystem
from cloudai.test_definitions.nemo_run import (
    Data,
    Log,
    LogCkpt,
    NeMoRunCmdArgs,
    NeMoRunTestDefinition,
    Trainer,
    TrainerStrategy,
)


@pytest.fixture
def test_run(tmp_path: Path) -> TestRun:
    tdef = NeMoRunTestDefinition(
        name="t1",
        description="desc1",
        test_template_name="tt",
        cmd_args=NeMoRunCmdArgs(docker_image_url="nvcr.io/nvidia/nemo:24.09", task="pretrain", recipe_name="llama_3b"),
        extra_env_vars={"TEST_VAR_1": "value1"},
        extra_cmd_args={"--extra_args": ""},
    )

    test = Test(test_definition=tdef, test_template=Mock())
    return TestRun(
        test=test,
        num_nodes=2,
        nodes=[],
        output_path=tmp_path / "output",
        name="test-job",
    )


@pytest.fixture
def cmd_gen_strategy(slurm_system: SlurmSystem) -> NeMoRunSlurmCommandGenStrategy:
    return NeMoRunSlurmCommandGenStrategy(slurm_system, {})


@pytest.mark.parametrize(
    "task,recipe_name,custom_recipe_path,expected_command",
    [
        ("fine_tune", "llama7_13b", None, ["nemo", "llm", "fine_tune", "--factory", "llama7_13b", "-y"]),
        ("custom", "llama7_13b", "/path/to/custom_recipe.py", ["python", "/path/to/custom_recipe.py", "--extra_args"]),
    ],
)
def test_generate_test_command(
    cmd_gen_strategy: NeMoRunSlurmCommandGenStrategy,
    test_run: TestRun,
    task: str,
    recipe_name: str,
    custom_recipe_path: Union[str, None],
    expected_command: list[str],
) -> None:
    cmd_args = NeMoRunCmdArgs(
        docker_image_url="nvcr.io/nvidia/nemo:24.09",
        task=task,
        recipe_name=recipe_name,
        custom_recipe_path=custom_recipe_path,
        trainer=Trainer(
            strategy=TrainerStrategy(tensor_model_parallel_size=2, virtual_pipeline_model_parallel_size=None),
        ),
        log=Log(ckpt=LogCkpt(save_last=False)),
        data=Data(micro_batch_size=1),
    )
    test_run.test.test_definition.cmd_args = cmd_args
    cmd = cmd_gen_strategy.generate_test_command(
        test_run.test.test_definition.extra_env_vars, test_run.test.test_definition.cmd_args.model_dump(), test_run
    )

    assert cmd is not None
    assert cmd[: len(expected_command)] == expected_command


def test_num_nodes(cmd_gen_strategy: NeMoRunSlurmCommandGenStrategy, test_run: TestRun) -> None:
    test_run.nodes = ["node[1-3]"]
    cmd = cmd_gen_strategy.generate_test_command(
        test_run.test.test_definition.extra_env_vars,
        test_run.test.test_definition.cmd_args.model_dump(),
        test_run,
    )

    num_nodes_param = next(p for p in cmd if "trainer.num_nodes" in p)
    assert num_nodes_param == "trainer.num_nodes=3"


@pytest.mark.parametrize(
    "task,custom_recipe_path,expected_mounts",
    [
        ("fine_tune", None, []),
        ("custom", "/path/to/custom_recipe.py", ["/path/to/custom_recipe.py:/path/to/custom_recipe.py"]),
    ],
)
def test_container_mounts(
    cmd_gen_strategy: NeMoRunSlurmCommandGenStrategy,
    test_run: TestRun,
    task: str,
    custom_recipe_path: Union[str, None],
    expected_mounts: list[str],
) -> None:
    test_run.test.test_definition.cmd_args.task = task
    test_run.test.test_definition.cmd_args.custom_recipe_path = custom_recipe_path

    mounts = cmd_gen_strategy._container_mounts(test_run)
    assert mounts == expected_mounts
