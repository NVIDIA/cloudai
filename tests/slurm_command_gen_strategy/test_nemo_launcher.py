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
from typing import cast
from unittest.mock import Mock, mock_open, patch

import pytest
from cloudai._core.test import Test
from cloudai._core.test_scenario import TestRun
from cloudai.schema.test_template.nemo_launcher.slurm_command_gen_strategy import (
    NeMoLauncherSlurmCommandGenStrategy,
)
from cloudai.schema.test_template.nemo_launcher.slurm_install_strategy import NeMoLauncherSlurmInstallStrategy
from cloudai.systems import SlurmSystem
from cloudai.test_definitions.nemo_launcher import NeMoLauncherCmdArgs, NeMoLauncherTestDefinition


@pytest.fixture
def nemo_test_run(tmp_path: Path) -> TestRun:
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


class TestNeMoLauncherSlurmCommandGenStrategy__GenExecCommand:
    @pytest.fixture
    def nemo_cmd_gen(self, slurm_system: SlurmSystem) -> NeMoLauncherSlurmCommandGenStrategy:
        cmd_args = {"test_arg": "test_value"}
        strategy = NeMoLauncherSlurmCommandGenStrategy(slurm_system, cmd_args)
        return strategy

    def test_generate_exec_command(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy, nemo_test_run: TestRun):
        cmd = nemo_cmd_gen.gen_exec_command(nemo_test_run)
        assert "TEST_VAR_1" in cmd
        assert "test_value" in cmd
        assert "extra_args" in cmd

        subdir = nemo_cmd_gen.system.install_path / NeMoLauncherSlurmInstallStrategy.SUBDIR_PATH
        assert f"{subdir}/nemo-venv/bin/python " in cmd

    def test_env_var_escaping(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy, nemo_test_run: TestRun):
        nemo_test_run.test.test_definition.extra_env_vars = {"TEST_VAR": "value,with,commas"}
        cmd = nemo_cmd_gen.gen_exec_command(nemo_test_run)

        assert "TEST_VAR=\\'value,with,commas\\'" in cmd

    def test_tokenizer_handled(
        self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy, nemo_test_run: TestRun, tmp_path: Path
    ):
        tokenizer_path = tmp_path / "tokenizer"
        tokenizer_path.touch()

        nemo_test_run.test.test_definition.extra_cmd_args = {f"training.model.tokenizer.model={tokenizer_path}": ""}

        cmd = nemo_cmd_gen.gen_exec_command(nemo_test_run)

        assert f"container_mounts=[{tokenizer_path}:{tokenizer_path}]" in cmd

    def test_reservation_handled(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy, nemo_test_run: TestRun):
        nemo_cmd_gen.system.extra_srun_args = "--reservation my-reservation"
        cmd = nemo_cmd_gen.gen_exec_command(nemo_test_run)

        assert "+cluster.reservation=my-reservation" in cmd

    def test_invalid_tokenizer_path(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy, nemo_test_run: TestRun):
        invalid_tokenizer_path = Path("/invalid/path/to/tokenizer")
        nemo_test_run.test.test_definition.extra_cmd_args = {
            f"training.model.tokenizer.model={invalid_tokenizer_path}": "",
        }

        with pytest.raises(
            ValueError,
            match=(
                r"The provided tokenizer path '/invalid/path/to/tokenizer' is not valid. Please review the test "
                r"schema file to ensure the tokenizer path is correct. If it contains a placeholder value, refer to "
                r"USER_GUIDE.md to download the tokenizer and update the schema file accordingly."
            ),
        ):
            nemo_cmd_gen.gen_exec_command(nemo_test_run)

    def test_account_in_command(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy, nemo_test_run: TestRun):
        nemo_cmd_gen.system.account = "test_account"
        cmd = nemo_cmd_gen.gen_exec_command(nemo_test_run)

        assert "cluster.account=test_account" in cmd
        assert "cluster.job_name_prefix=test_account-cloudai.nemo:" in cmd

    def test_no_account_in_command(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy, nemo_test_run: TestRun):
        nemo_cmd_gen.system.account = None
        cmd = nemo_cmd_gen.gen_exec_command(nemo_test_run)

        assert "cluster.account" not in cmd
        assert "cluster.job_name_prefix" not in cmd

    def test_gpus_per_node_value(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy, nemo_test_run: TestRun):
        nemo_cmd_gen.system.gpus_per_node = 4
        cmd = nemo_cmd_gen.gen_exec_command(nemo_test_run)

        assert "cluster.gpus_per_node=4" in cmd

        nemo_cmd_gen.system.gpus_per_node = None
        cmd = nemo_cmd_gen.gen_exec_command(nemo_test_run)

        assert "cluster.gpus_per_node=null" in cmd

    def test_data_impl_mock_skips_checks(
        self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy, nemo_test_run: TestRun
    ):
        tdef: NeMoLauncherTestDefinition = cast(NeMoLauncherTestDefinition, nemo_test_run.test.test_definition)
        tdef.extra_cmd_args = {"data_dir": "DATA_DIR"}
        cmd = nemo_cmd_gen.gen_exec_command(nemo_test_run)
        assert "data_dir=DATA_DIR" in cmd

    def test_data_dir_and_data_prefix_validation(
        self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy, nemo_test_run: TestRun
    ):
        tdef: NeMoLauncherTestDefinition = cast(NeMoLauncherTestDefinition, nemo_test_run.test.test_definition)
        tdef.cmd_args.training.model.data.data_impl = "not_mock"
        tdef.cmd_args.training.model.data.data_prefix = "[]"
        tdef.extra_cmd_args = {"data_dir": "DATA_DIR"}

        with pytest.raises(ValueError, match="The 'data_prefix' field of the NeMo launcher test is missing or empty."):
            nemo_cmd_gen.gen_exec_command(nemo_test_run)

    @patch("pathlib.Path.open", new_callable=mock_open)
    def test_log_command_to_file(
        self, mock_file, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy, nemo_test_run: TestRun, tmp_path: Path
    ):
        nemo_test_run.output_path = tmp_path / "output_dir"
        nemo_test_run.output_path.mkdir()
        nemo_test_run.num_nodes = 1

        nemo_cmd_gen.gen_exec_command(nemo_test_run)

        written_content = mock_file().write.call_args[0][0]

        assert " \\\n " in written_content, "Command should contain line breaks when written to the file"
        assert "python" in written_content, "Logged command should start with 'python'"
        assert "TEST_VAR_1=value1" in written_content, "Logged command should contain environment variables"
        assert "training.trainer.num_nodes=1" in written_content, "Command should contain the number of nodes"

    def test_no_line_breaks_in_executed_command(
        self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy, nemo_test_run: TestRun, tmp_path: Path
    ):
        nemo_test_run.output_path = tmp_path / "output_dir"
        nemo_test_run.output_path.mkdir()

        cmd = nemo_cmd_gen.gen_exec_command(nemo_test_run)

        assert "\n" not in cmd, "Executed command should not contain line breaks"
