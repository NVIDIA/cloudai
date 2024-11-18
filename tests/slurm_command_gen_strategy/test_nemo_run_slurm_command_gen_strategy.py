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
from unittest.mock import Mock

import pytest

from cloudai._core.test import Test
from cloudai._core.test_scenario import TestRun
from cloudai.schema.test_template.nemo_run.slurm_command_gen_strategy import NeMoRunSlurmCommandGenStrategy
from cloudai.systems import SlurmSystem
from cloudai.test_definitions.nemo_run import NeMoRunCmdArgs, NeMoRunTestDefinition


class TestNeMoRunSlurmCommandGenStrategy:
    @pytest.fixture
    def test_run(self, tmp_path: Path) -> TestRun:
        tdef = NeMoRunTestDefinition(
            name="t1",
            description="desc1",
            test_template_name="tt",
            cmd_args=NeMoRunCmdArgs(
                docker_image_url="nvcr.io/nvidia/nemo:24.09", task="pretrain", recipe_name="llama_3b"
            ),
            extra_env_vars={"TEST_VAR_1": "value1"},
            extra_cmd_args={"extra_args": ""},
        )

        test = Test(test_definition=tdef, test_template=Mock())
        tr = TestRun(
            test=test,
            num_nodes=2,
            nodes=[],
            output_path=tmp_path / "output",
            name="test-job",
        )

        return tr

    @pytest.fixture
    def cmd_gen_strategy(self, slurm_system: SlurmSystem) -> NeMoRunSlurmCommandGenStrategy:
        return NeMoRunSlurmCommandGenStrategy(slurm_system, {})

    @pytest.mark.parametrize(
        "cmd_args, expected_cmd",
        [
            (
                {"docker_image_url": "nvcr.io/nvidia/nemo:24.09", "task": "fine_tune", "recipe_name": "llama7_13b"},
                ["nemo", "llm", "fine_tune", "--factory", "llama7_13b", "-y", "trainer.num_nodes=2", "extra_args"],
            ),
        ],
    )
    def test_generate_test_command(
        self,
        cmd_gen_strategy: NeMoRunSlurmCommandGenStrategy,
        test_run: TestRun,
        cmd_args: dict,
        expected_cmd: list,
    ) -> None:
        test_run.test.test_definition.cmd_args = NeMoRunCmdArgs(**cmd_args)

        cmd = cmd_gen_strategy.generate_test_command(
            test_run.test.test_definition.extra_env_vars,
            test_run.test.test_definition.cmd_args.model_dump(),
            test_run,
        )
        assert cmd == expected_cmd, f"Expected command {expected_cmd}, but got {cmd}"

    @pytest.mark.parametrize(
        "cmd_args, expected_exception",
        [
            ({"docker_image_url": "nvcr.io/nvidia/nemo:24.09", "recipe_name": "llama7_13b"}, ValueError),
            ({"task": "fine_tune"}, ValueError),
        ],
    )
    def test_generate_test_command_exceptions(
        self,
        cmd_gen_strategy: NeMoRunSlurmCommandGenStrategy,
        test_run: TestRun,
        cmd_args: dict,
        expected_exception: type,
    ) -> None:
        test_run.test.test_definition.cmd_args = NeMoRunCmdArgs(**cmd_args)

        with pytest.raises(expected_exception):
            cmd_gen_strategy.generate_test_command(
                test_run.test.test_definition.extra_env_vars,
                test_run.test.test_definition.cmd_args.model_dump(),
                test_run,
            )
