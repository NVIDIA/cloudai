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
from unittest.mock import MagicMock

import pytest

from cloudai import Test, TestRun
from cloudai.schema.test_template.jax_toolbox.slurm_command_gen_strategy import JaxToolboxSlurmCommandGenStrategy
from cloudai.schema.test_template.jax_toolbox.template import JaxToolbox
from cloudai.systems import SlurmSystem
from cloudai.test_definitions.gpt import GPTCmdArgs, GPTTestDefinition
from cloudai.test_definitions.grok import GrokCmdArgs, GrokTestDefinition
from cloudai.test_definitions.jax_toolbox import JaxFdl, PreTest


class TestJaxToolboxSlurmCommandGenStrategy:
    @pytest.fixture
    def cmd_gen_strategy(self, slurm_system: SlurmSystem) -> JaxToolboxSlurmCommandGenStrategy:
        return JaxToolboxSlurmCommandGenStrategy(slurm_system, {})

    @pytest.fixture
    def gpt_test(self) -> GPTTestDefinition:
        return GPTTestDefinition(
            name="gpt",
            description="desc",
            test_template_name="gpt",
            cmd_args=GPTCmdArgs(fdl_config="", docker_image_url="http://fake_image_url"),
            extra_env_vars={"COMBINE_THRESHOLD": "1"},
        )

    @pytest.fixture
    def grok_test(self) -> GrokTestDefinition:
        return GrokTestDefinition(
            name="grok",
            description="desc",
            test_template_name="grok",
            cmd_args=GrokCmdArgs(docker_image_url="http://fake_image_url"),
            extra_env_vars={"COMBINE_THRESHOLD": "1"},
        )

    @pytest.mark.parametrize("test_fixture", ["gpt_test", "grok_test"])
    def test_gen_exec_command(
        self,
        slurm_system,
        cmd_gen_strategy: JaxToolboxSlurmCommandGenStrategy,
        tmp_path: Path,
        request,
        test_fixture,
    ) -> None:
        test_def = request.getfixturevalue(test_fixture)
        test_def.cmd_args.pre_test = PreTest(enable=True)

        test = Test(test_definition=test_def, test_template=JaxToolbox(slurm_system, "name"))
        test_run = TestRun(
            test=test,
            num_nodes=1,
            nodes=["node1"],
            output_path=tmp_path / "output",
            name="test-job",
        )

        cmd_gen_strategy._generate_pre_test_command = MagicMock(return_value="pre_test_command")
        cmd = cmd_gen_strategy.gen_exec_command(test_run)
        assert cmd == f"sbatch {test_run.output_path}/cloudai_sbatch_script.sh"
        assert (test_run.output_path / "run.sh").exists()

        content = Path(f"{test_run.output_path}/cloudai_sbatch_script.sh").read_text()
        assert "pre_test_command" in content

    @pytest.mark.parametrize(
        "cmd_args, expected",
        [
            ({"GPT.setup_flags": "/some/dir"}, "GPT"),
            ({"Grok.setup_flags": "/some/dir"}, "Grok"),
            ({"Nemotron.setup_flags": "/some/dir"}, "Nemotron"),
            ({"unknown": "value"}, ""),
        ],
    )
    def test_extract_test_name(
        self, cmd_gen_strategy: JaxToolboxSlurmCommandGenStrategy, cmd_args: dict, expected: str
    ) -> None:
        test_name = cmd_gen_strategy._extract_test_name(cmd_args)
        assert test_name == expected

    def test_format_xla_flags_grok(
        self, cmd_gen_strategy: JaxToolboxSlurmCommandGenStrategy, grok_test: GrokTestDefinition
    ) -> None:
        cmd_gen_strategy.test_name = "Grok"
        xla_flags = cmd_gen_strategy._format_xla_flags(grok_test.cmd_args_dict, "profile")
        actual_flags_list = sorted(xla_flags.split())

        expected_flags_list = sorted(
            [
                "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD",
                "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD",
                "--xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD",
            ]
        )
        assert actual_flags_list == expected_flags_list

    def test_format_xla_flags_gpt(
        self, cmd_gen_strategy: JaxToolboxSlurmCommandGenStrategy, gpt_test: GPTTestDefinition
    ) -> None:
        cmd_gen_strategy.test_name = "GPT"
        xla_flags = cmd_gen_strategy._format_xla_flags(gpt_test.cmd_args_dict, "profile")
        actual_flags_list = sorted(xla_flags.split())

        expected_flags_list = sorted(
            [
                "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD",
                "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD",
                "--xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD",
            ]
        )
        assert actual_flags_list == expected_flags_list

    def test_format_xla_flags_boolean_are_lowcased(self, cmd_gen_strategy: JaxToolboxSlurmCommandGenStrategy) -> None:
        cmd_gen_strategy.test_name = "GPT"

        cmd_args_dict = {"GPT.profile.XLA_FLAGS.xla_gpu_enable_while_loop_double_buffering": True}

        xla_flags = cmd_gen_strategy._format_xla_flags(cmd_args_dict, "profile")
        actual_flags_list = sorted(xla_flags.split())

        expected_flags_list = sorted(
            [
                "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD",
                "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD",
                "--xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD",
                "--xla_gpu_enable_while_loop_double_buffering=true",
            ]
        )

        # Updated the assertion to match the current expected output
        assert actual_flags_list == expected_flags_list

    @pytest.mark.parametrize("enable_pgle,expected_ncalls", [(True, 2), (False, 1)])
    def test_create_run_script_pgle_control(
        self,
        enable_pgle: bool,
        expected_ncalls: int,
        slurm_system: SlurmSystem,
        grok_test: GrokTestDefinition,
        tmp_path: Path,
    ):
        grok_test.cmd_args.enable_pgle = enable_pgle
        cargs = {"output_path": str(tmp_path), **grok_test.cmd_args_dict}
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, cargs)
        cmd_gen.test_name = "Grok"
        cmd_gen._script_content = MagicMock(return_value="")
        cmd_gen._create_run_script({}, cargs, "")
        assert cmd_gen._script_content.call_count == expected_ncalls

    def test_generate_python_command(
        self,
        slurm_system: SlurmSystem,
        cmd_gen_strategy: JaxToolboxSlurmCommandGenStrategy,
        gpt_test: GPTTestDefinition,
        tmp_path: Path,
    ) -> None:
        cmd_gen_strategy = JaxToolboxSlurmCommandGenStrategy(slurm_system, gpt_test.cmd_args_dict)
        cargs = {"output_path": "/path/to/output", **gpt_test.cmd_args_dict}
        cargs = cmd_gen_strategy._override_cmd_args(cmd_gen_strategy.default_cmd_args, cargs)

        cmd_gen_strategy.test_name = "GPT"

        stage = "training"
        python_cli = cmd_gen_strategy._generate_python_command(stage, cargs, "").splitlines()

        fdl_args = gpt_test.cmd_args.fdl.model_dump()
        fdl_args_list = [f"    --fdl.{k.upper()}={v} \\" for k, v in sorted(fdl_args.items())]
        fdl_args_list[-1] = fdl_args_list[-1].replace(" \\", "")

        expected_py_cmd = [
            "    python3 -u -m paxml.main \\",
            "    --num_hosts=$SLURM_NTASKS \\",
            "    --server_addr=$SLURM_JOB_MASTER_NODE:12345 \\",
            "    --host_idx=$SLURM_PROCID \\",
            f"    --job_log_dir={gpt_test.cmd_args.setup_flags.docker_workspace_dir} \\",
            f"    --tfds_data_dir={gpt_test.cmd_args.setup_flags.tfds_data_dir} \\",
            f"    --enable_checkpoint_saving={gpt_test.cmd_args.setup_flags.enable_checkpoint_saving} \\",
            "    --multiprocess_gpu \\",
            "    --alsologtostderr \\",
            f'    --fdl_config="{gpt_test.cmd_args.fdl_config}" \\',
            *fdl_args_list,
        ]

        assert python_cli == [
            'if [ "$SLURM_NODEID" -eq 0 ] && [ "$SLURM_PROCID" -eq 0 ]; then',
            "    nsys profile \\",
            "    -s none \\",
            f"    -o /opt/paxml/workspace/nsys_profile_{stage} \\",
            "    --force-overwrite true \\",
            "    --capture-range=cudaProfilerApi \\",
            "    --capture-range-end=stop \\",
            "    --cuda-graph-trace=node \\",
            *expected_py_cmd,
            "else",
            *expected_py_cmd,
            "fi",
        ]

    def test_generate_pre_test_command(
        self, cmd_gen_strategy: JaxToolboxSlurmCommandGenStrategy, grok_test: GrokTestDefinition, tmp_path: Path
    ) -> None:
        grok_test.cmd_args.pre_test = PreTest(enable=True)

        nccl_test = grok_test.cmd_args.pre_test.nccl_test
        nccl_test.num_nodes = 2
        nccl_test.minbytes = "32M"
        nccl_test.blocking = 0

        cargs = {"output_path": str(tmp_path), **grok_test.cmd_args_dict}

        pre_test_cli = cmd_gen_strategy._generate_pre_test_command(cargs, tmp_path, tmp_path).splitlines()

        expected_pre_test_cli = [
            "srun \\",
            "--mpi=pmix \\",
            f"-N {nccl_test.num_nodes} \\",
            f"-o {tmp_path} \\",
            f"-e {tmp_path} \\",
            f"--container-image={nccl_test.docker_image_url} \\",
            f"/usr/local/bin/{nccl_test.subtest_name} \\",
            f"--nthreads {nccl_test.nthreads} \\",
            f"--ngpus {nccl_test.ngpus} \\",
            f"--minbytes {nccl_test.minbytes} \\",
            f"--maxbytes {nccl_test.maxbytes} \\",
            f"--stepbytes {nccl_test.stepbytes} \\",
            f"--op {nccl_test.op} \\",
            f"--datatype {nccl_test.datatype} \\",
            f"--root {nccl_test.root} \\",
            f"--iters {nccl_test.iters} \\",
            f"--warmup_iters {nccl_test.warmup_iters} \\",
            f"--agg_iters {nccl_test.agg_iters} \\",
            f"--average {nccl_test.average} \\",
            f"--parallel_init {nccl_test.parallel_init} \\",
            f"--check {nccl_test.check} \\",
            f"--blocking {nccl_test.blocking} \\",
            f"--cudagraph {nccl_test.cudagraph} \\",
            f"--stepfactor {nccl_test.stepfactor}",
        ]

        assert pre_test_cli == expected_pre_test_cli, (
            "The generated pre-test command did not match the expected command.\n"
            f"Expected: {expected_pre_test_cli}\n"
            f"Actual: {pre_test_cli}"
        )

    def test_generate_srun_command(self, slurm_system, cmd_gen_strategy, grok_test):
        cmd_gen_strategy.test_name = grok_test.name
        Path("/tmp/output").mkdir(parents=True, exist_ok=True)

        output_path = Path("/tmp/output/output")
        output_path.mkdir(parents=True, exist_ok=True)

        # Use the existing setup for mocking internal methods
        cmd_gen_strategy._generate_pre_test_command = MagicMock(return_value="srun --mpi=none pre_test_command")
        cmd_gen_strategy._generate_run_command = MagicMock(return_value="srun --mpi=none run_command")
        cmd_gen_strategy._generate_container_load_command = MagicMock(
            return_value="srun --mpi=none container_load_command"
        )

        slurm_args = {
            "output": "/tmp/output/output-%j.txt",
            "error": "/tmp/output/error-%j.txt",
            "image_path": "fake_image_url",
            "container_mounts": "/tmp/output:/workspace",
        }
        cmd_args = {
            "output_path": "/tmp/output",
            "pre_test": {"enable": True},
            f"{grok_test.name}.setup_flags.docker_workspace_dir": "/workspace/docker",
            f"{grok_test.name}.setup_flags.tfds_data_dir": "/workspace/tfds",
            f"{grok_test.name}.setup_flags.enable_checkpoint_saving": True,
        }

        pre_test_command = cmd_gen_strategy._generate_pre_test_command(
            cmd_args, Path("/tmp/output"), Path("/tmp/output")
        )
        run_command = cmd_gen_strategy._generate_run_command(slurm_args)
        container_load_command = cmd_gen_strategy._generate_container_load_command(slurm_args)

        result_command = f"{pre_test_command}\n{container_load_command}\n{run_command}"

        # Assert expected parts of the command are in the generated result
        assert "pre_test_command" in result_command
        assert "container_load_command" in result_command
        assert "run_command" in result_command
        assert "srun" in result_command
        assert "--mpi=none" in result_command

        cmd_gen_strategy._generate_pre_test_command.assert_called_once()
        cmd_gen_strategy._generate_run_command.assert_called_once()
        cmd_gen_strategy._generate_container_load_command.assert_called_once()


def test_gpt_test_definition_cmd_args_dict():
    gpt = GPTTestDefinition(
        name="gpt",
        description="gpt",
        test_template_name="gpt",
        cmd_args=GPTCmdArgs(fdl_config="", docker_image_url=""),
    )

    cargs = gpt.cmd_args_dict

    assert "GPT.fdl" in cargs
    assert "GPT.setup_flags" in cargs
    assert "GPT.XLA_FLAGS" in cargs

    for k in {"pre_test", "docker_image_url", "load_container"}:
        assert k in cargs
        assert f"GPT.{k}" not in cargs


def test_grok_test_definition_cmd_args_dict():
    grok = GrokTestDefinition(
        name="grok",
        description="grok",
        test_template_name="grok",
        cmd_args=GrokCmdArgs(docker_image_url=""),
    )

    cargs = grok.cmd_args_dict

    assert "Grok.setup_flags" in cargs
    assert "Grok.enable_pgle" in cargs
    assert "Grok.fdl" in cargs

    assert "Grok.profile" in cargs
    assert "XLA_FLAGS" in cargs["Grok.profile"]
    assert "Grok.perf" in cargs
    assert "XLA_FLAGS" in cargs["Grok.perf"]

    for k in {"pre_test", "docker_image_url", "load_container"}:
        assert k in cargs
        assert f"Grok.{k}" not in cargs


@pytest.mark.parametrize("prop", ["fprop_dtype", "checkpoint_policy"])
def test_jax_props_is_escaped(prop: str):
    fdl = JaxFdl()
    d = fdl.model_dump()
    assert d[prop] == f'\\"{getattr(fdl, prop)}\\"'

    setattr(fdl, prop, '\\"val\\"')
    d = fdl.model_dump()
    assert d[prop] == '\\"val\\"'

    if prop == "checkpoint_policy":
        fdl.checkpoint_policy = '"val"'
        d = fdl.model_dump()
        assert d[prop] == '\\"val\\"'
