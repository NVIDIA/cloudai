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

import re
from pathlib import Path
from typing import List, cast
from unittest.mock import mock_open, patch

import pytest

from cloudai import Test, TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.nemo_launcher import (
    NeMoLauncherCmdArgs,
    NeMoLauncherSlurmCommandGenStrategy,
    NeMoLauncherTestDefinition,
)


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
        (tmp_path / "repo").mkdir()
        (tmp_path / "venv").mkdir()
        tdef.python_executable.git_repo.installed_path = tmp_path / "repo"
        tdef.python_executable.venv_path = tmp_path / "venv"

        test = Test(test_definition=tdef)
        tr = TestRun(
            test=test,
            num_nodes=2,
            nodes=[],
            output_path=tmp_path / "output",
            name="test-job",
        )

        return tr

    @pytest.fixture
    def cmd_gen_strategy(self, slurm_system: SlurmSystem, test_run: TestRun) -> NeMoLauncherSlurmCommandGenStrategy:
        return NeMoLauncherSlurmCommandGenStrategy(slurm_system, test_run)

    @pytest.mark.parametrize(
        "expected_content, nodes",
        [
            (
                [
                    'TEST_VAR_1="value1"',
                    '+env_vars.TEST_VAR_1="value1"',
                    'stages=["training"]',
                    "cluster.gpus_per_node=8",
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
                    "training.trainer.max_steps=20",
                    "training.trainer.num_nodes=2",
                    "training.trainer.val_check_interval=10",
                    "training=gpt3/40b_improved",
                    "+cluster.nodelist=\\'node1,node2\\'",
                ],
                ["node1", "node2"],
            ),
        ],
    )
    def test_generate_exec_command(
        self,
        cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy,
        expected_content: List[str],
        nodes: List[str],
    ) -> None:
        cmd_gen_strategy.test_run.nodes = nodes
        cmd = cmd_gen_strategy.gen_exec_command()

        for content in expected_content:
            assert any(content in part for part in cmd.split())
        assert "training.run.name=" in cmd
        assert "extra_args" in cmd
        assert "base_results_dir=" in cmd
        assert "launcher_scripts_path=" in cmd
        tdef: NeMoLauncherTestDefinition = cast(
            NeMoLauncherTestDefinition, cmd_gen_strategy.test_run.test.test_definition
        )
        assert f"container={tdef.docker_image.url}" in cmd

    def test_tokenizer_handling(self, cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy, tmp_path: Path) -> None:
        tokenizer_path = tmp_path / "tokenizer"
        tokenizer_path.touch()

        cmd_gen_strategy.test_run.test.test_definition.extra_cmd_args = {
            f"training.model.tokenizer.model={tokenizer_path}": ""
        }
        cmd = cmd_gen_strategy.gen_exec_command()

        assert f'container_mounts=["{tokenizer_path}:{tokenizer_path}"]' in cmd

    @pytest.mark.parametrize(
        "extra_srun_args, expected_reservation",
        [
            ("--reservation my-reservation", "+cluster.reservation=my-reservation"),
            ("", None),
        ],
    )
    def test_reservation_handling(
        self, cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy, extra_srun_args: str, expected_reservation: str
    ) -> None:
        cmd_gen_strategy.system.extra_srun_args = extra_srun_args
        cmd = cmd_gen_strategy.gen_exec_command()

        if expected_reservation:
            assert expected_reservation in cmd
        else:
            assert "+cluster.reservation" not in cmd

    def test_invalid_tokenizer_path(self, cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy) -> None:
        invalid_tokenizer_path = Path("/invalid/path/to/tokenizer")
        cmd_gen_strategy.test_run.test.test_definition.extra_cmd_args = {
            f"training.model.tokenizer.model={invalid_tokenizer_path}": "",
        }

        with pytest.raises(ValueError, match=r"The provided tokenizer path '/invalid/path/to/tokenizer' is not valid"):
            cmd_gen_strategy.gen_exec_command()

    @pytest.mark.parametrize(
        "account, expected_prefix_pattern",
        [
            ("test_account", r"test_account-cloudai\.nemo_\d{8}_\d{6}:"),
            (None, r"\d{8}_\d{6}:"),
        ],
    )
    def test_account_in_command(
        self,
        cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy,
        test_run: TestRun,
        account: str,
        expected_prefix_pattern: str,
    ) -> None:
        cmd_gen_strategy.system.account = account
        cmd = cmd_gen_strategy.gen_exec_command()

        if account:
            assert f"cluster.account={account}" in cmd

        match = re.search(r"cluster.job_name_prefix=([\w\-\._:]+)", cmd)
        assert match, f"Expected cluster.job_name_prefix in command, but not found. Command: {cmd}"

        job_name_prefix = match.group(1)
        assert re.match(expected_prefix_pattern, job_name_prefix), f"Unexpected job_name_prefix: {job_name_prefix}"

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
        gpus_per_node: int,
        expected_gpus: str,
    ) -> None:
        cmd_gen_strategy.system.gpus_per_node = gpus_per_node
        cmd = cmd_gen_strategy.gen_exec_command()

        assert expected_gpus in cmd

    def test_data_prefix_validation(self, cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy) -> None:
        tdef: NeMoLauncherTestDefinition = cast(
            NeMoLauncherTestDefinition, cmd_gen_strategy.test_run.test.test_definition
        )
        tdef.cmd_args.training.model.data.data_impl = "not_mock"
        tdef.cmd_args.training.model.data.data_prefix = "[]"

        with pytest.raises(ValueError, match=r"The 'data_prefix' field of the NeMo launcher test is missing or empty."):
            cmd_gen_strategy.gen_exec_command()

    @patch("pathlib.Path.open", new_callable=mock_open)
    def test_log_command_to_file(
        self, mock_file, cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy, tmp_path: Path
    ) -> None:
        cmd_gen_strategy.test_run.output_path = tmp_path / "output_dir"
        cmd_gen_strategy.test_run.output_path.mkdir()

        repo_path = (tmp_path / "repo").relative_to(tmp_path)
        tdef: NeMoLauncherTestDefinition = cast(
            NeMoLauncherTestDefinition, cmd_gen_strategy.test_run.test.test_definition
        )
        tdef.python_executable.git_repo.installed_path = repo_path
        tdef.python_executable.venv_path = repo_path.parent / f"{repo_path.name}-venv"
        cmd_gen_strategy.gen_exec_command()

        written_content = mock_file().write.call_args[0][0]

        assert " \\\n " in written_content, "Command should contain line breaks when written to the file"
        assert "python" in written_content, "Logged command should start with 'python'"
        assert 'TEST_VAR_1="value1"' in written_content, "Logged command should contain environment variables"
        assert "training.trainer.num_nodes=2" in written_content, "Command should contain the number of nodes"

        assert str((tdef.python_executable.venv_path / "bin" / "python").absolute()) in written_content
        assert (
            f"launcher_scripts_path={(repo_path / tdef.cmd_args.launcher_script).parent.absolute()} " in written_content
        )

    def test_container_mounts_with_nccl_topo_file(self, cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy) -> None:
        nccl_topo_file_path = "/opt/topo.toml"
        cmd_gen_strategy.test_run.test.test_definition.extra_env_vars["NCCL_TOPO_FILE"] = nccl_topo_file_path

        cmd = cmd_gen_strategy.gen_exec_command()

        expected_mount = f'container_mounts=["{nccl_topo_file_path}:{nccl_topo_file_path}"]'
        assert expected_mount in cmd

    def test_env_vars_with_spaces(self, cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy) -> None:
        env: dict[str, str | list[str]] = {"VAR1": "value with spaces", "VAR2": r"$(cmd \$vv| cmd)"}
        cmd = cmd_gen_strategy._gen_env_vars_str(env)

        assert cmd == 'VAR1="value with spaces" \\\nVAR2="$(cmd \\$vv| cmd)" \\\n'

    @pytest.mark.parametrize(
        "args,expected",
        [
            ({"env_vars.VAR1": "value1"}, '+env_vars.VAR1="value1"'),
            ({"env_vars.VAR1": "value1,v2"}, "+env_vars.VAR1=\\'value1,v2\\'"),
        ],
    )
    def test_generate_cmd_args_str_handles_env_vars(
        self, cmd_gen_strategy: NeMoLauncherSlurmCommandGenStrategy, args: dict[str, str | list[str]], expected: str
    ) -> None:
        cmd = cmd_gen_strategy._generate_cmd_args_str(args, [])
        assert cmd == expected
