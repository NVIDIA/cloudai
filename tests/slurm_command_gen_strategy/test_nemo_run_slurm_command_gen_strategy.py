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

import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from cloudai._core.test import Test
from cloudai._core.test_scenario import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.nemo_run import (
    Data,
    NeMoRunCmdArgs,
    NeMoRunSlurmCommandGenStrategy,
    NeMoRunTestDefinition,
    Trainer,
    TrainerStrategy,
)


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
            num_nodes=1,
            nodes=[],
            output_path=tmp_path / "output",
            name="test-job",
        )

        return tr

    @pytest.fixture
    def cmd_gen_strategy(self, slurm_system: SlurmSystem) -> NeMoRunSlurmCommandGenStrategy:
        return NeMoRunSlurmCommandGenStrategy(slurm_system, {})

    def test_generate_test_command(self, cmd_gen_strategy: NeMoRunSlurmCommandGenStrategy, test_run: TestRun) -> None:
        cmd_args = NeMoRunCmdArgs(
            docker_image_url="nvcr.io/nvidia/nemo:24.09",
            task="fine_tune",
            recipe_name="llama7_13b",
            trainer=Trainer(
                strategy=TrainerStrategy(tensor_model_parallel_size=2, virtual_pipeline_model_parallel_size=None),
            ),
            data=Data(micro_batch_size=1),
        )
        test_run.test.test_definition.cmd_args = cmd_args

        recipe_name = cmd_gen_strategy._validate_recipe_name(cmd_args.recipe_name)

        cmd = cmd_gen_strategy.generate_test_command(
            test_run.test.test_definition.extra_env_vars, test_run.test.test_definition.cmd_args.model_dump(), test_run
        )
        assert cmd is not None
        assert cmd[:5] == [
            "python",
            f"/cloudai_install/{cmd_gen_strategy._run_script(test_run).name}",
            "--factory",
            recipe_name,
            "-y",
        ]
        assert (
            f"trainer.strategy.tensor_model_parallel_size={cmd_args.trainer.strategy.tensor_model_parallel_size}" in cmd
        )
        assert f"data.micro_batch_size={cmd_args.data.micro_batch_size}" in cmd

    def test_num_nodes(self, cmd_gen_strategy: NeMoRunSlurmCommandGenStrategy, test_run: TestRun) -> None:
        test_run.nodes = ["node1"]
        cmd_args_dict = test_run.test.test_definition.cmd_args.model_dump()
        cmd_args_dict["trainer"]["num_nodes"] = len(test_run.nodes)

        cmd = cmd_gen_strategy.generate_test_command(
            test_run.test.test_definition.extra_env_vars,
            cmd_args_dict,
            test_run,
        )

        assert any("trainer.num_nodes=1" in param for param in cmd)

    def test_trainer_num_nodes_greater_than_num_nodes(
        self, cmd_gen_strategy: NeMoRunSlurmCommandGenStrategy, test_run: TestRun, caplog
    ) -> None:
        test_run.num_nodes = 1
        cmd_args = NeMoRunCmdArgs(
            docker_image_url="nvcr.io/nvidia/nemo:24.09",
            task="fine_tune",
            recipe_name="llama7_13b",
            trainer=Trainer(
                num_nodes=4,
            ),
        )
        test_run.test.test_definition.cmd_args = cmd_args

        with caplog.at_level(logging.WARNING), pytest.raises(SystemExit) as excinfo:
            cmd_gen_strategy.generate_test_command(
                test_run.test.test_definition.extra_env_vars,
                test_run.test.test_definition.cmd_args.model_dump(),
                test_run,
            )
        assert excinfo.value.code == 1
        assert "Mismatch in num_nodes" in caplog.text
