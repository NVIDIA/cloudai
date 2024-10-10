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
from typing import List, cast
from unittest.mock import Mock, mock_open, patch

import pytest
from cloudai._core.test import Test
from cloudai._core.test_scenario import TestRun
from cloudai.schema.test_template.nemo_launcher.slurm_command_gen_strategy import NeMoLauncherSlurmCommandGenStrategy
from cloudai.systems import SlurmSystem
from cloudai.test_definitions.nemo_launcher import NeMoLauncherCmdArgs, NeMoLauncherTestDefinition


class TestNeMoLauncherSlurmCommandGenStrategy:
    @pytest.fixture
    def test_run(self, tmp_path: Path) -> TestRun:
        tdef = NeMoLauncherTestDefinition(
            name="t1",
            description="desc1",
            test_template_name="tt",
            cmd_args=NeMoLauncherCmdArgs(),
            extra_env_vars={"TEST_VAR_1": "value1"},
            extra_cmd_args={"extra_args": ""},
        )
        test = Test(
            test_definition=tdef,
            test_template=Mock(),
        )
        tr = TestRun(
            test=test,
            num_nodes=2,
            nodes=[],
            output_path=tmp_path / "output",
            name="test-job",
        )
        return tr

    @pytest.fixture
    def cmd_gen_strategy(self, slurm_system: SlurmSystem) -> NeMoLauncherSlurmCommandGenStrategy:
        return NeMoLauncherSlurmCommandGenStrategy(slurm_system, {})

    @pytest.mark.parametrize(
        "expected_content",
        [
            [
                "TEST_VAR_1=value1",
                "+env_vars.TEST_VAR_1=value1",
                'stages=["training"]',
                "cluster.gpus_per_node=null",
                "cluster.partition=main",
                "numa_mapping.enable=True",
                "training.exp_manager.create_checkpoint_callback=False",
                "training.model.data.data_impl=mock",
                "training.model.data.data_prefix=[]",
                "training.model.global_batch_size=128",
                "training.model.micro_batch_size=2",
                "training.model.pipeline_model_parallel_size=4",
                "training.model.tensor_model_parallel_size=4",
                "training.run.time_limit=3:00:00",
                "training.trainer.enable_checkpointing=False",
                "training.trainer.log_every_n_steps=1",
                "training.trainer.max_steps=400",
                "training.trainer.num_nodes=2",
                "training.trainer.val_check_interval=100",
                "training=gpt3/40b_improved",
            ]
        ],
    )
    def test_generate_exec_command(
        self,
        cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy,
        test_run: TestRun,
        expected_content: List[str],
    ) -> None:
        cmd = cmd_gen_strategy.gen_exec_command(test_run)
        for content in expected_content:
            assert any(content in part for part in cmd.split())
        assert "training.run.name=" in cmd
        assert "extra_args" in cmd
        assert "base_results_dir=" in cmd
        assert "launcher_scripts_path=" in cmd

    def test_tokenizer_handling(
        self, cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy, test_run: TestRun, tmp_path: Path
    ) -> None:
        tokenizer_path = tmp_path / "tokenizer"
        tokenizer_path.touch()

        test_run.test.test_definition.extra_cmd_args = {f"training.model.tokenizer.model={tokenizer_path}": ""}
        cmd = cmd_gen_strategy.gen_exec_command(test_run)

        assert f"container_mounts=[{tokenizer_path}:{tokenizer_path}]" in cmd

    @pytest.mark.parametrize(
        "extra_srun_args, expected_reservation",
        [
            ("--reservation my-reservation", "+cluster.reservation=my-reservation"),
            ("", None),
        ],
    )
    def test_reservation_handling(
        self,
        cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy,
        test_run: TestRun,
        extra_srun_args: str,
        expected_reservation: str,
    ) -> None:
        cmd_gen_strategy.system.extra_srun_args = extra_srun_args
        cmd = cmd_gen_strategy.gen_exec_command(test_run)

        if expected_reservation:
            assert expected_reservation in cmd
        else:
            assert "+cluster.reservation" not in cmd

    def test_invalid_tokenizer_path(
        self, cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy, test_run: TestRun
    ) -> None:
        invalid_tokenizer_path = Path("/invalid/path/to/tokenizer")
        test_run.test.test_definition.extra_cmd_args = {
            f"training.model.tokenizer.model={invalid_tokenizer_path}": "",
        }

        with pytest.raises(ValueError, match=r"The provided tokenizer path '/invalid/path/to/tokenizer' is not valid"):
            cmd_gen_strategy.gen_exec_command(test_run)

    @pytest.mark.parametrize(
        "account, expected_prefix",
        [
            ("test_account", "test_account-cloudai.nemo:"),
            (None, None),
        ],
    )
    def test_account_in_command(
        self,
        cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy,
        test_run: TestRun,
        account: str,
        expected_prefix: str,
    ) -> None:
        cmd_gen_strategy.system.account = account
        cmd = cmd_gen_strategy.gen_exec_command(test_run)

        if expected_prefix:
            assert f"cluster.account={account}" in cmd
            assert f"cluster.job_name_prefix={expected_prefix}" in cmd
        else:
            assert "cluster.account" not in cmd
            assert "cluster.job_name_prefix" not in cmd

    @pytest.mark.parametrize(
        "gpus_per_node, expected_gpus",
        [
            (4, "cluster.gpus_per_node=4"),
            (None, "cluster.gpus_per_node=null"),
        ],
    )
    def test_gpus_per_node_value(
        self,
        cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy,
        test_run: TestRun,
        gpus_per_node: int,
        expected_gpus: str,
    ) -> None:
        cmd_gen_strategy.system.gpus_per_node = gpus_per_node
        cmd = cmd_gen_strategy.gen_exec_command(test_run)

        assert expected_gpus in cmd

    def test_data_prefix_validation(
        self, cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy, test_run: TestRun
    ) -> None:
        tdef: NeMoLauncherTestDefinition = cast(NeMoLauncherTestDefinition, test_run.test.test_definition)
        tdef.cmd_args.training.model.data.data_impl = "not_mock"
        tdef.cmd_args.training.model.data.data_prefix = "[]"

        with pytest.raises(ValueError, match="The 'data_prefix' field of the NeMo launcher test is missing or empty."):
            cmd_gen_strategy.gen_exec_command(test_run)

    @patch("pathlib.Path.open", new_callable=mock_open)
    def test_log_command_to_file(
        self, mock_file, cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy, test_run: TestRun, tmp_path: Path
    ) -> None:
        test_run.output_path = tmp_path / "output_dir"
        test_run.output_path.mkdir()

        cmd_gen_strategy.gen_exec_command(test_run)

        written_content = mock_file().write.call_args[0][0]

        assert " \\\n " in written_content, "Command should contain line breaks when written to the file"
        assert "python" in written_content, "Logged command should start with 'python'"
        assert "TEST_VAR_1=value1" in written_content, "Logged command should contain environment variables"
        assert "training.trainer.num_nodes=2" in written_content, "Command should contain the number of nodes"

    def test_no_line_breaks_in_executed_command(
        self, cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy, test_run: TestRun, tmp_path: Path
    ) -> None:
        test_run.output_path = tmp_path / "output_dir"
        test_run.output_path.mkdir()

        cmd = cmd_gen_strategy.gen_exec_command(test_run)

        assert "\n" not in cmd, "Executed command should not contain line breaks"
