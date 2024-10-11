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
from unittest.mock import MagicMock, Mock, patch

import pytest
from cloudai import Test, TestRun
from cloudai.schema.test_template.jax_toolbox.slurm_command_gen_strategy import JaxToolboxSlurmCommandGenStrategy
from cloudai.schema.test_template.jax_toolbox.template import JaxToolbox
from cloudai.systems import SlurmSystem
from cloudai.systems.slurm import SlurmNodeState
from cloudai.systems.slurm.slurm_system import SlurmPartition
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
from cloudai.test_definitions.gpt import GPTCmdArgs, GPTTestDefinition
from cloudai.test_definitions.grok import GrokCmdArgs, GrokTestDefinition
from cloudai.test_definitions.jax_toolbox import JaxFdl, PreTest


@pytest.fixture
def slurm_system(tmp_path: Path) -> SlurmSystem:
    slurm_system = SlurmSystem(
        name="TestSystem",
        install_path=tmp_path / "install",
        output_path=tmp_path / "output",
        default_partition="main",
        partitions=[
            SlurmPartition(name="main", nodes=["node[1-4]"]),
        ],
        mpi="fake-mpi",
    )
    for node in slurm_system.partitions[0].slurm_nodes:
        node.state = SlurmNodeState.IDLE
    Path(slurm_system.install_path).mkdir()
    Path(slurm_system.output_path).mkdir()
    return slurm_system


@pytest.fixture
def strategy_fixture(slurm_system: SlurmSystem) -> SlurmCommandGenStrategy:
    cmd_args = {"test_arg": "test_value"}
    strategy = SlurmCommandGenStrategy(slurm_system, cmd_args)
    return strategy


@pytest.fixture
def jax_strategy_fixture() -> JaxToolboxSlurmCommandGenStrategy:
    # Mock the SlurmSystem and other dependencies
    mock_slurm_system = Mock()
    cmd_args = {"test_arg": "test_value"}
    mock_slurm_system.install_path = "/mock/install/path"

    # Use patch to mock the __init__ method of JaxToolboxSlurmCommandGenStrategy
    with patch.object(JaxToolboxSlurmCommandGenStrategy, "__init__", lambda self, _, __: None):
        strategy = JaxToolboxSlurmCommandGenStrategy(mock_slurm_system, cmd_args)
        # Manually set attributes needed for the tests
        strategy.cmd_args = cmd_args
        strategy.default_cmd_args = cmd_args
        return strategy


@pytest.fixture
def gpt_test() -> GPTTestDefinition:
    return GPTTestDefinition(
        name="gpt",
        description="desc",
        test_template_name="gpt",
        cmd_args=GPTCmdArgs(fdl_config="", docker_image_url=""),
        extra_env_vars={"COMBINE_THRESHOLD": "1"},  # it is always set in Test TOMLs
    )


@pytest.fixture
def grok_test() -> GrokTestDefinition:
    return GrokTestDefinition(
        name="grok",
        description="desc",
        test_template_name="grok",
        cmd_args=GrokCmdArgs(docker_image_url=""),
        extra_env_vars={"COMBINE_THRESHOLD": "1"},  # it is always set in Test TOMLs
    )


class TestGenerateSrunCommand__CmdGeneration:
    def test_generate_full_srun_command_with_pre_test(self, slurm_system: SlurmSystem, gpt_test: GPTTestDefinition):
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, gpt_test.cmd_args_dict)
        cmd_gen._create_run_script = MagicMock()
        cmd_gen._generate_pre_test_command = MagicMock(return_value="pre_test_command")
        cmd_gen._generate_pre_test_check_command = MagicMock(return_value="pre_test_check_command")

        slurm_args = {
            "output": "output.txt",
            "error": "error.txt",
            "image_path": "image_path",
            "container_mounts": "container_mounts",
            "container_name": "cont",
        }
        gpt_test.cmd_args.pre_test = PreTest()
        gpt_test.cmd_args.output_path = "/path/to/output"
        cargs = {"output_path": "/path/to/output", **gpt_test.cmd_args_dict}
        result = cmd_gen.generate_srun_command(slurm_args, {}, cargs, "")
        assert "pre_test_command" in result
        assert "pre_test_check_command" in result
        assert "--mpi=none" in result
        assert "--container-mounts=" + slurm_args["container_mounts"] in result
        assert f"-o {slurm_args['output']}" in result
        assert f"-e {slurm_args['error']}" in result
        assert "--container-name=" + slurm_args["container_name"] in result
        assert "/opt/paxml/workspace/run.sh" in result

    def test_generate_full_srun_command_without_pre_test(self, slurm_system: SlurmSystem, gpt_test: GPTTestDefinition):
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, gpt_test.cmd_args_dict)
        cmd_gen._create_run_script = MagicMock()
        cmd_gen._generate_pre_test_command = MagicMock(return_value="pre_test_command")
        cmd_gen._generate_pre_test_check_command = MagicMock(return_value="pre_test_check_command")

        slurm_args = {
            "output": "output.txt",
            "error": "error.txt",
            "image_path": "image_path",
            "container_mounts": "container_mounts",
            "container_name": "cont",
        }
        gpt_test.cmd_args.pre_test = PreTest(enable=False)
        cargs = {"output_path": "/path/to/output", **gpt_test.cmd_args_dict}
        result = cmd_gen.generate_srun_command(slurm_args, {}, cargs, "")

        assert "pre_test_command" not in result
        assert "pre_test_check_command" not in result
        assert "--mpi=none" in result
        assert f"--container-mounts={slurm_args['container_mounts']}" in result
        assert "--container-name=" + slurm_args.get("container_name", "") in result
        assert f"-o {slurm_args['output']}" in result
        assert f"-e {slurm_args['error']}" in result

    def test_gen_exec_command(self, slurm_system: SlurmSystem, tmp_path: Path, gpt_test: GPTTestDefinition):
        test = Test(test_definition=gpt_test, test_template=JaxToolbox(slurm_system, "name"))

        test_run = TestRun(
            test=test,
            num_nodes=1,
            nodes=["node1"],
            output_path=tmp_path / "output",
            name="test-job",
        )

        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, test_run.test.cmd_args)

        cmd = cmd_gen.gen_exec_command(test_run)

        assert cmd == f"sbatch {test_run.output_path}/cloudai_sbatch_script.sh"
        assert (test_run.output_path / "run.sh").exists()

    def test_generate_pre_test_command(self, slurm_system: SlurmSystem, gpt_test: GPTTestDefinition, tmp_path: Path):
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, gpt_test.cmd_args_dict)
        cargs = {"output_path": "/path/to/output", **gpt_test.cmd_args_dict}
        pre_test_cli = cmd_gen._generate_pre_test_command(cargs, tmp_path, tmp_path).splitlines()
        nccl_test = gpt_test.cmd_args.pre_test.nccl_test
        assert pre_test_cli == [
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

    def test_generate_python_command(self, slurm_system: SlurmSystem, gpt_test: GPTTestDefinition, tmp_path: Path):
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, gpt_test.cmd_args_dict)
        cargs = {"output_path": "/path/to/output", **gpt_test.cmd_args_dict}
        cargs = cmd_gen._override_cmd_args(cmd_gen.default_cmd_args, cargs)
        cmd_gen.test_name = "GPT"
        stage = "stage"
        python_cli = cmd_gen._generate_python_command(stage, cargs, "").splitlines()

        fdl_args = gpt_test.cmd_args.fdl.model_dump()
        fdl_args_list = []
        for k, v in sorted(fdl_args.items()):
            fdl_args_list.append(f"    --fdl.{k.upper()}={v} \\")
        fdl_args_list[-1] = fdl_args_list[-1].replace(" \\", "")
        py_cmd = [
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
            *py_cmd,
            "else",
            *py_cmd,
            "fi",
        ]

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
        cmd_args = grok_test.cmd_args_dict
        cmd_args["output_path"] = str(tmp_path)
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, cmd_args)
        cmd_gen.test_name = "Grok"
        cmd_gen._script_content = Mock(return_value="")
        cmd_gen._create_run_script({}, cmd_args, "")
        assert (tmp_path / "run.sh").exists()
        assert cmd_gen._script_content.call_count == expected_ncalls


class TestJaxToolboxSlurmCommandGenStrategy__ExtractTestName:
    @pytest.mark.parametrize(
        "cmd_args, expected",
        [
            ({"Grok.setup_flags": "/some/dir"}, "Grok"),
            ({"GPT.setup_flags": "/some/dir"}, "GPT"),
            ({"Nemotron.setup_flags": "/some/dir"}, "Nemotron"),
            ({"unknown": "value"}, ""),
        ],
    )
    def test_extract_test_name(
        self, cmd_args: dict, expected: str, jax_strategy_fixture: JaxToolboxSlurmCommandGenStrategy
    ):
        test_name = jax_strategy_fixture._extract_test_name(cmd_args)
        assert test_name == expected

    def test_format_xla_flags_grok(self, grok_test: GrokTestDefinition, slurm_system: SlurmSystem):
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, grok_test.cmd_args_dict)
        cmd_gen.test_name = "Grok"

        xla_flags = cmd_gen._format_xla_flags(grok_test.cmd_args_dict, "profile")

        actual_flags_list = sorted(xla_flags.split())

        profile_xlas = [f"--{k}={v}" for k, v in grok_test.cmd_args.profile.model_dump().items()]
        profile_xlas = [x.replace("=False", "=false") for x in profile_xlas]
        profile_xlas = [x.replace("=True", "=true") for x in profile_xlas]
        expected_flags_list = sorted(
            [
                "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD",
                "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD",
                "--xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD",
                *profile_xlas,
            ]
        )
        assert actual_flags_list == expected_flags_list

        xla_flags = cmd_gen._format_xla_flags(grok_test.cmd_args_dict, "perf")

        actual_flags_list = sorted(xla_flags.split())

        perf_xlas = [f"--{k}={v}" for k, v in grok_test.cmd_args.perf.model_dump().items() if k.startswith("xla_")]
        perf_xlas = [x.replace("=False", "=false") for x in perf_xlas]
        perf_xlas = [x.replace("=True", "=true") for x in perf_xlas]
        expected_flags_list = sorted(
            [
                "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD",
                "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD",
                "--xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD",
                *perf_xlas,
            ]
        )
        assert actual_flags_list == expected_flags_list

    def test_format_xla_flags_gpt(self, gpt_test: GPTTestDefinition, slurm_system: SlurmSystem):
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, gpt_test.cmd_args_dict)
        cmd_gen.test_name = "GPT"
        xla_flags = cmd_gen._format_xla_flags(gpt_test.cmd_args_dict, "profile")

        actual_flags_list = sorted(xla_flags.split())
        expected_flags_list = sorted(
            "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD "
            "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD "
            "--xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD".split()
        )
        assert actual_flags_list == expected_flags_list

    def test_format_xla_flags_boolean_are_lowcased(self, slurm_system: SlurmSystem):
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, {})
        cmd_gen.test_name = "GPT"

        xla_flags = cmd_gen._format_xla_flags(
            {"GPT.profile": {"XLA_FLAGS": {"xla_gpu_enable_while_loop_double_buffering": True}}}, "profile"
        ).split()
        assert len(xla_flags) == 4
        assert xla_flags[-2] == "--xla_gpu_enable_while_loop_double_buffering=true"


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
