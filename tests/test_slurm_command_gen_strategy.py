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
from unittest.mock import Mock, patch

import pytest
from cloudai.schema.test_template.jax_toolbox.slurm_command_gen_strategy import JaxToolboxSlurmCommandGenStrategy
from cloudai.schema.test_template.nccl_test.slurm_command_gen_strategy import NcclTestSlurmCommandGenStrategy
from cloudai.schema.test_template.nemo_launcher.slurm_command_gen_strategy import (
    NeMoLauncherSlurmCommandGenStrategy,
)
from cloudai.systems import SlurmSystem
from cloudai.systems.slurm import SlurmNode, SlurmNodeState
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy


@pytest.fixture
def slurm_system(tmp_path: Path) -> SlurmSystem:
    slurm_system = SlurmSystem(
        name="TestSystem",
        install_path=str(tmp_path / "install"),
        output_path=str(tmp_path / "output"),
        default_partition="main",
        extra_srun_args="",
        partitions={
            "main": [
                SlurmNode(name="node1", partition="main", state=SlurmNodeState.IDLE),
                SlurmNode(name="node2", partition="main", state=SlurmNodeState.IDLE),
                SlurmNode(name="node3", partition="main", state=SlurmNodeState.IDLE),
                SlurmNode(name="node4", partition="main", state=SlurmNodeState.IDLE),
            ]
        },
        mpi="fake-mpi",
    )
    Path(slurm_system.install_path).mkdir()
    Path(slurm_system.output_path).mkdir()
    return slurm_system


@pytest.fixture
def strategy_fixture(slurm_system: SlurmSystem) -> SlurmCommandGenStrategy:
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    strategy = SlurmCommandGenStrategy(slurm_system, env_vars, cmd_args)
    return strategy


@pytest.fixture
def jax_strategy_fixture() -> JaxToolboxSlurmCommandGenStrategy:
    # Mock the SlurmSystem and other dependencies
    mock_slurm_system = Mock()
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}

    # Use patch to mock the __init__ method of JaxToolboxSlurmCommandGenStrategy
    with patch.object(JaxToolboxSlurmCommandGenStrategy, "__init__", lambda self, _, __, ___: None):
        strategy = JaxToolboxSlurmCommandGenStrategy(mock_slurm_system, env_vars, cmd_args)
        # Manually set attributes needed for the tests
        strategy.env_vars = env_vars
        strategy.cmd_args = cmd_args
        strategy.default_env_vars = env_vars
        strategy.default_cmd_args = cmd_args
        return strategy


def test_filename_generation(strategy_fixture: SlurmCommandGenStrategy, tmp_path: Path):
    args = {"job_name": "test_job", "num_nodes": 2, "partition": "test_partition", "node_list_str": "node1,node2"}
    env_vars_str = "export TEST_VAR=VALUE"
    srun_command = "srun --test test_arg"
    output_path = str(tmp_path)

    sbatch_command = strategy_fixture._write_sbatch_script(args, env_vars_str, srun_command, output_path)
    filepath_from_command = sbatch_command.split()[-1]

    # Check that the file exists at the specified path
    assert tmp_path.joinpath("cloudai_sbatch_script.sh").exists()

    # Read the file and check the contents
    with open(filepath_from_command, "r") as file:
        file_contents = file.read()
    assert "test_job" in file_contents
    assert "node1,node2" in file_contents
    assert "srun --test test_arg" in file_contents

    # Check the correctness of the sbatch command format
    assert sbatch_command == f"sbatch {filepath_from_command}"


def test_num_nodes_and_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    nodes = ["node1", "node2"]
    num_nodes = 3

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)

    assert slurm_args["num_nodes"] == len(nodes)


def test_only_num_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    nodes = []
    num_nodes = 3

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)

    assert slurm_args["num_nodes"] == num_nodes


def test_only_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    nodes = ["node1", "node2"]
    num_nodes = 0

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)

    assert slurm_args["num_nodes"] == len(nodes)


class TestGenerateSrunCommand__CmdGeneration:
    def test_generate_test_command(self, strategy_fixture: SlurmCommandGenStrategy):
        test_command = strategy_fixture.generate_test_command({}, {}, {}, "")
        assert test_command == []

    def test_generate_srun_command(self, strategy_fixture: SlurmCommandGenStrategy):
        srun_command = strategy_fixture.generate_srun_command({}, {}, {}, "")
        assert srun_command == ["srun", f"--mpi={strategy_fixture.slurm_system.mpi}"]

    def test_generate_srun_command_with_extra_args(self, strategy_fixture: SlurmCommandGenStrategy):
        strategy_fixture.slurm_system.extra_srun_args = "--extra-args value"
        srun_command = strategy_fixture.generate_srun_command({}, {}, {}, "")
        assert srun_command == ["srun", f"--mpi={strategy_fixture.slurm_system.mpi}", "--extra-args value"]

    def test_generate_srun_command_with_container_image(self, strategy_fixture: SlurmCommandGenStrategy):
        slurm_args = {"image_path": "fake_image_path"}
        srun_command = strategy_fixture.generate_srun_command(slurm_args, {}, {}, "")
        assert srun_command == [
            "srun",
            f"--mpi={strategy_fixture.slurm_system.mpi}",
            "--container-image=fake_image_path",
        ]

    def test_generate_srun_command_with_container_image_and_mounts(self, strategy_fixture: SlurmCommandGenStrategy):
        slurm_args = {"image_path": "fake_image_path", "container_mounts": "fake_mounts"}
        srun_command = strategy_fixture.generate_srun_command(slurm_args, {}, {}, "")
        assert srun_command == [
            "srun",
            f"--mpi={strategy_fixture.slurm_system.mpi}",
            "--container-image=fake_image_path",
            "--container-mounts=fake_mounts",
        ]

    def test_generate_srun_empty_str(self, strategy_fixture: SlurmCommandGenStrategy):
        slurm_args = {"image_path": "", "container_mounts": ""}
        srun_command = strategy_fixture.generate_srun_command(slurm_args, {}, {}, "")
        assert srun_command == ["srun", f"--mpi={strategy_fixture.slurm_system.mpi}"]

        slurm_args = {"image_path": "fake", "container_mounts": ""}
        srun_command = strategy_fixture.generate_srun_command(slurm_args, {}, {}, "")
        assert srun_command == ["srun", f"--mpi={strategy_fixture.slurm_system.mpi}", "--container-image=fake"]

    def test_generate_full_srun_command(self, strategy_fixture: SlurmCommandGenStrategy):
        strategy_fixture.generate_srun_command = lambda *_, **__: ["srun", "--test", "test_arg"]
        strategy_fixture.generate_test_command = lambda *_, **__: ["test_command"]

        full_srun_command = strategy_fixture.generate_full_srun_command({}, {}, {}, "")
        assert full_srun_command == " \\\n".join(["srun", "--test", "test_arg", "test_command"])


class TestJaxToolboxSlurmCommandGenStrategy__ExtractTestName:
    def test_extract_test_name_grok(self, jax_strategy_fixture: JaxToolboxSlurmCommandGenStrategy):
        cmd_args = {"Grok.setup_flags.docker_workspace_dir": "/some/dir", "Grok.some_other_flag": "value"}
        test_name = jax_strategy_fixture._extract_test_name(cmd_args)
        assert test_name == "Grok"

    def test_extract_test_name_gpt(self, jax_strategy_fixture: JaxToolboxSlurmCommandGenStrategy):
        cmd_args = {"GPT.setup_flags.docker_workspace_dir": "/some/dir", "GPT.some_other_flag": "value"}
        test_name = jax_strategy_fixture._extract_test_name(cmd_args)
        assert test_name == "GPT"

    def test_extract_test_name_none(self, jax_strategy_fixture: JaxToolboxSlurmCommandGenStrategy):
        cmd_args = {"some_other_flag": "value", "another_flag": "value"}
        test_name = jax_strategy_fixture._extract_test_name(cmd_args)
        assert test_name == ""

    def test_format_xla_flags_grok(self, jax_strategy_fixture: JaxToolboxSlurmCommandGenStrategy):
        jax_strategy_fixture.test_name = "Grok"
        cmd_args = {
            "Grok.XLA_FLAGS.some_flag": "value",
            "Grok.XLA_FLAGS.another_flag": "another_value",
        }
        xla_flags = jax_strategy_fixture._format_xla_flags(cmd_args)
        expected_flags = (
            "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD "
            "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD "
            "--some_flag=value --another_flag=another_value"
        )
        assert xla_flags == expected_flags

    def test_format_xla_flags_gpt(self, jax_strategy_fixture: JaxToolboxSlurmCommandGenStrategy):
        jax_strategy_fixture.test_name = "GPT"
        cmd_args = {
            "GPT.XLA_FLAGS.some_flag": "value",
            "GPT.XLA_FLAGS.another_flag": "another_value",
        }
        xla_flags = jax_strategy_fixture._format_xla_flags(cmd_args)
        expected_flags = (
            "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD "
            "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD "
            "--xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD "
            "--some_flag=value --another_flag=another_value"
        )
        assert xla_flags == expected_flags

    def test_format_xla_flags_common(self, jax_strategy_fixture: JaxToolboxSlurmCommandGenStrategy):
        jax_strategy_fixture.test_name = "SomeTest"
        cmd_args = {
            "common.XLA_FLAGS.some_flag": "value",
            "common.XLA_FLAGS.another_flag": "another_value",
        }
        xla_flags = jax_strategy_fixture._format_xla_flags(cmd_args)
        expected_flags = "--some_flag=value --another_flag=another_value"
        assert xla_flags == expected_flags

    def test_format_xla_flags_boolean(self, jax_strategy_fixture: JaxToolboxSlurmCommandGenStrategy):
        jax_strategy_fixture.test_name = "Grok"
        cmd_args = {
            "Grok.XLA_FLAGS.some_flag": "value",
            "Grok.XLA_FLAGS.xla_gpu_simplify_all_fp_conversions": True,
        }
        xla_flags = jax_strategy_fixture._format_xla_flags(cmd_args)
        expected_flags = (
            "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD "
            "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD "
            "--some_flag=value --xla_gpu_simplify_all_fp_conversions"
        )
        assert xla_flags == expected_flags

    def test_combine_fdl_flags(self, jax_strategy_fixture: JaxToolboxSlurmCommandGenStrategy):
        cmd_args = {
            "common.fdl.some_flag": "value",
            "common.fdl.another_flag": "another_value",
            "Grok.fdl.test_flag": "test_value",
        }
        combined_flags = jax_strategy_fixture._combine_fdl_flags(cmd_args, "Grok")
        expected_flags = {
            "some_flag": "value",
            "another_flag": "another_value",
            "test_flag": "test_value",
        }
        assert combined_flags == expected_flags

    def test_combine_fdl_flags_override(self, jax_strategy_fixture: JaxToolboxSlurmCommandGenStrategy):
        cmd_args = {
            "common.fdl.some_flag": "value",
            "Grok.fdl.some_flag": "overridden_value",
        }
        combined_flags = jax_strategy_fixture._combine_fdl_flags(cmd_args, "Grok")
        expected_flags = {
            "some_flag": "overridden_value",
        }
        assert combined_flags == expected_flags

    def test_get_fdl_config_value(self, jax_strategy_fixture: JaxToolboxSlurmCommandGenStrategy):
        cmd_args = {
            "Grok.fdl_config": "config_value",
        }
        jax_strategy_fixture.test_name = "Grok"
        config_value = jax_strategy_fixture._get_fdl_config_value(cmd_args)
        assert config_value == "config_value"

    def test_get_fdl_config_value_missing(self, jax_strategy_fixture: JaxToolboxSlurmCommandGenStrategy):
        cmd_args = {}
        jax_strategy_fixture.test_name = "Grok"
        config_value = jax_strategy_fixture._get_fdl_config_value(cmd_args)
        assert config_value is None


class TestNeMoLauncherSlurmCommandGenStrategy__GenExecCommand:
    @pytest.fixture
    def nemo_cmd_gen(self, slurm_system: SlurmSystem) -> NeMoLauncherSlurmCommandGenStrategy:
        env_vars = {"TEST_VAR": "VALUE"}
        cmd_args = {"test_arg": "test_value"}
        strategy = NeMoLauncherSlurmCommandGenStrategy(slurm_system, env_vars, cmd_args)
        return strategy

    def test_extra_env_vars_added(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy):
        extra_env_vars = {"TEST_VAR_1": "value1", "TEST_VAR_2": "value2"}
        cmd_args = {
            "docker_image_url": "fake",
            "repository_url": "fake",
            "repository_commit_hash": "fake",
        }
        cmd = nemo_cmd_gen.gen_exec_command(
            env_vars={},
            cmd_args=cmd_args,
            extra_env_vars=extra_env_vars,
            extra_cmd_args="",
            output_path="",
            num_nodes=1,
            nodes=[],
        )

        for k, v in extra_env_vars.items():
            assert f"{k}={v}" in cmd

    def test_env_var_escaping(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy):
        extra_env_vars = {"TEST_VAR": "value,with,commas"}
        cmd_args = {
            "docker_image_url": "fake",
            "repository_url": "fake",
            "repository_commit_hash": "fake",
        }
        cmd = nemo_cmd_gen.gen_exec_command(
            env_vars={},
            cmd_args=cmd_args,
            extra_env_vars=extra_env_vars,
            extra_cmd_args="",
            output_path="",
            num_nodes=1,
            nodes=[],
        )

        assert "TEST_VAR=\\'value,with,commas\\'" in cmd

    def test_tokenizer_handled(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy, tmp_path: Path):
        extra_env_vars = {"TEST_VAR_1": "value1"}
        cmd_args = {
            "docker_image_url": "fake",
            "repository_url": "fake",
            "repository_commit_hash": "fake",
        }
        tokenizer_path = tmp_path / "tokenizer"
        tokenizer_path.touch()

        cmd = nemo_cmd_gen.gen_exec_command(
            env_vars={},
            cmd_args=cmd_args,
            extra_env_vars=extra_env_vars,
            extra_cmd_args=f"training.model.tokenizer.model={tokenizer_path}",
            output_path="",
            num_nodes=1,
            nodes=[],
        )

        assert f"container_mounts=[{tokenizer_path}:{tokenizer_path}]" in cmd

    def test_reservation_handled(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy):
        extra_env_vars = {"TEST_VAR_1": "value1"}
        cmd_args = {
            "docker_image_url": "fake",
            "repository_url": "fake",
            "repository_commit_hash": "fake",
        }
        nemo_cmd_gen.slurm_system.extra_srun_args = "--reservation my-reservation"
        cmd = nemo_cmd_gen.gen_exec_command(
            env_vars={},
            cmd_args=cmd_args,
            extra_cmd_args="",
            extra_env_vars=extra_env_vars,
            output_path="",
            num_nodes=1,
            nodes=[],
        )

        assert "+cluster.reservation=my-reservation" in cmd

    def test_invalid_tokenizer_path(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy):
        extra_env_vars = {"TEST_VAR_1": "value1"}
        cmd_args = {
            "docker_image_url": "fake",
            "repository_url": "fake",
            "repository_commit_hash": "fake",
        }
        invalid_tokenizer_path = "/invalid/path/to/tokenizer"

        with pytest.raises(
            ValueError,
            match=(
                r"The provided tokenizer path '/invalid/path/to/tokenizer' is not valid. Please review the test "
                r"schema file to ensure the tokenizer path is correct. If it contains a placeholder value, refer to "
                r"USER_GUIDE.md to download the tokenizer and update the schema file accordingly."
            ),
        ):
            nemo_cmd_gen.gen_exec_command(
                env_vars={},
                cmd_args=cmd_args,
                extra_env_vars=extra_env_vars,
                extra_cmd_args=f"training.model.tokenizer.model={invalid_tokenizer_path}",
                output_path="",
                num_nodes=1,
                nodes=[],
            )


class TestWriteSbatchScript:
    MANDATORY_ARGS = {
        "job_name": "test_job",
        "num_nodes": 2,
        "partition": "test_partition",
        "node_list_str": "node1,node2",
    }

    def setup_method(self):
        self.env_vars_str = "export TEST_VAR=VALUE"
        self.srun_command = "srun --test test_arg"

    def assert_positional_lines(self, lines: list[str]):
        assert lines[0] == "#!/bin/bash"
        assert lines[-6] == ""
        assert lines[-5] == "export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)"
        assert lines[-4] == ""
        assert lines[-3] == self.env_vars_str
        assert lines[-2] == ""
        assert lines[-1] == self.srun_command

    @pytest.mark.parametrize("missing_arg", ["job_name", "num_nodes", "partition", "node_list_str"])
    def test_raises_on_missing_args(self, missing_arg: str, strategy_fixture: SlurmCommandGenStrategy, tmp_path: Path):
        args = self.MANDATORY_ARGS.copy()
        del args[missing_arg]

        with pytest.raises(KeyError) as exc_info:
            strategy_fixture._write_sbatch_script(args, self.env_vars_str, self.srun_command, str(tmp_path))
        assert f"KeyError('{missing_arg}')" in str(exc_info)

    def test_only_mandatory_args(self, strategy_fixture: SlurmCommandGenStrategy, tmp_path: Path):
        sbatch_command = strategy_fixture._write_sbatch_script(
            self.MANDATORY_ARGS, self.env_vars_str, self.srun_command, str(tmp_path)
        )

        filepath_from_command = sbatch_command.split()[-1]
        assert sbatch_command == f"sbatch {filepath_from_command}"
        assert tmp_path.joinpath("cloudai_sbatch_script.sh").exists()

        with open(filepath_from_command, "r") as file:
            file_contents = file.read()

        lines = file_contents.splitlines()
        assert len(lines) == 13

        self.assert_positional_lines(lines)

        assert f"#SBATCH --job-name={self.MANDATORY_ARGS['job_name']}" in file_contents
        assert f"#SBATCH -N {self.MANDATORY_ARGS['num_nodes']}" in file_contents
        assert f"#SBATCH --partition={self.MANDATORY_ARGS['partition']}" in file_contents
        assert f"#SBATCH --nodelist={self.MANDATORY_ARGS['node_list_str']}" in file_contents
        assert f"#SBATCH --output={tmp_path / 'stdout.txt'}" in file_contents
        assert f"#SBATCH --error={tmp_path / 'stderr.txt'}" in file_contents

    @pytest.mark.parametrize(
        "arg, arg_value, expected_str",
        [
            ("account", "test_account", "#SBATCH --account=test_account"),
            ("distribution", "block", "#SBATCH --distribution=block"),
            ("gpus_per_node", 2, "#SBATCH --gpus-per-node=2"),
            ("ntasks_per_node", 2, "#SBATCH --ntasks-per-node=2"),
            ("time_limit", "00:30:00", "#SBATCH --time=00:30:00"),
        ],
    )
    def test_extra_args(
        self, arg: str, arg_value: str, expected_str: str, strategy_fixture: SlurmCommandGenStrategy, tmp_path: Path
    ):
        args = self.MANDATORY_ARGS.copy()
        args[arg] = arg_value

        sbatch_command = strategy_fixture._write_sbatch_script(
            args, self.env_vars_str, self.srun_command, str(tmp_path)
        )

        filepath_from_command = sbatch_command.split()[-1]
        with open(filepath_from_command, "r") as file:
            file_contents = file.read()

        self.assert_positional_lines(file_contents.splitlines())
        assert expected_str in file_contents

    @pytest.mark.parametrize("add_arg", ["output", "error"])
    def test_disable_output_and_error(self, add_arg: str, strategy_fixture: SlurmCommandGenStrategy, tmp_path: Path):
        args = self.MANDATORY_ARGS.copy()
        args[add_arg] = "fake"

        sbatch_command = strategy_fixture._write_sbatch_script(
            args, self.env_vars_str, self.srun_command, str(tmp_path)
        )

        filepath_from_command = sbatch_command.split()[-1]
        with open(filepath_from_command, "r") as file:
            file_contents = file.read()

        self.assert_positional_lines(file_contents.splitlines())
        assert f"--{add_arg}=" not in file_contents


class TestNCCLSlurmCommandGen:
    def get_cmd(self, slurm_system: SlurmSystem, slurm_args: dict, cmd_args: dict) -> str:
        return NcclTestSlurmCommandGenStrategy(slurm_system, {}, {}).generate_full_srun_command(
            slurm_args, {}, cmd_args, ""
        )

    def test_only_mandatory(self, slurm_system: SlurmSystem) -> None:
        slurm_args = {"image_path": "fake_image_path"}
        cmd_args = {"subtest_name": "fake_subtest_name"}
        cmd = self.get_cmd(slurm_system, slurm_args, cmd_args)
        assert cmd == " \\\n".join(
            [
                "srun",
                f"--mpi={slurm_system.mpi}",
                f"--container-image={slurm_args['image_path']}",
                f"/usr/local/bin/{cmd_args['subtest_name']}",
            ]
        )

    def test_with_container_mounts(self, slurm_system: SlurmSystem) -> None:
        slurm_args = {"image_path": "fake_image_path", "container_mounts": "fake_mounts"}
        cmd_args = {"subtest_name": "fake_subtest_name"}
        cmd = self.get_cmd(slurm_system, slurm_args, cmd_args)
        assert cmd == " \\\n".join(
            [
                "srun",
                f"--mpi={slurm_system.mpi}",
                f"--container-image={slurm_args['image_path']}",
                f"--container-mounts={slurm_args['container_mounts']}",
                f"/usr/local/bin/{cmd_args['subtest_name']}",
            ]
        )
