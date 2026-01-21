# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
from unittest.mock import Mock, patch

import pytest
import toml

from cloudai.cli import setup_logging
from cloudai.cli.handlers import handle_dry_run_and_run
from cloudai.core import CommandGenStrategy, GitRepo, TestDefinition, TestRun, TestScenario
from cloudai.models.scenario import TestRunDetails
from cloudai.systems.slurm import SlurmCommandGenStrategy, SlurmRunner, SlurmSystem
from cloudai.workloads.ai_dynamo import (
    AIDynamoArgs,
    AIDynamoCmdArgs,
    AIDynamoTestDefinition,
    DecodeWorkerArgs,
    GenAIPerfArgs,
    PrefillWorkerArgs,
)
from cloudai.workloads.ddlb import DDLBCmdArgs, DDLBTestDefinition
from cloudai.workloads.deepep import (
    DeepEPCmdArgs,
    DeepEPTestDefinition,
)
from cloudai.workloads.jax_toolbox import (
    GPTCmdArgs,
    GPTTestDefinition,
    GrokCmdArgs,
    GrokTestDefinition,
)
from cloudai.workloads.megatron_run import (
    MegatronRunCmdArgs,
    MegatronRunTestDefinition,
)
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition
from cloudai.workloads.nemo_launcher import (
    NeMoLauncherCmdArgs,
    NeMoLauncherSlurmCommandGenStrategy,
    NeMoLauncherTestDefinition,
)
from cloudai.workloads.nemo_run import NeMoRunCmdArgs, NeMoRunTestDefinition
from cloudai.workloads.nixl_bench import NIXLBenchCmdArgs, NIXLBenchTestDefinition
from cloudai.workloads.nixl_kvbench import NIXLKVBenchCmdArgs, NIXLKVBenchTestDefinition
from cloudai.workloads.nixl_perftest import NixlPerftestCmdArgs, NixlPerftestTestDefinition
from cloudai.workloads.osu_bench import OSUBenchCmdArgs, OSUBenchTestDefinition
from cloudai.workloads.sleep import SleepCmdArgs, SleepTestDefinition
from cloudai.workloads.slurm_container import (
    SlurmContainerCmdArgs,
    SlurmContainerTestDefinition,
)
from cloudai.workloads.triton_inference import (
    TritonInferenceCmdArgs,
    TritonInferenceTestDefinition,
)
from cloudai.workloads.ucc_test import UCCCmdArgs, UCCTestDefinition

SLURM_TEST_SCENARIOS = [
    {"path": Path("conf/common/test_scenario/sleep.toml"), "expected_dirs_number": 4, "log_file": "sleep_debug.log"},
    {
        "path": Path("conf/common/test_scenario/ucc_test.toml"),
        "expected_dirs_number": 4,
        "log_file": "ucc_test_debug.log",
    },
]


class TestInDryRun:
    @pytest.fixture(scope="class", params=SLURM_TEST_SCENARIOS)
    def do_dry_run(self, tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest) -> tuple[Path, dict]:
        tmp_path = tmp_path_factory.mktemp("dry_run")
        scenario = request.param

        test_scenario_path = scenario["path"]
        log_file = scenario.get("log_file", ".")
        log_file_path = tmp_path / log_file

        setup_logging(log_file_path, "DEBUG")
        args = argparse.Namespace(
            mode="dry-run",
            system_config=Path("conf/common/system/example_slurm_cluster.toml"),
            test_templates_dir=Path("conf/common/test_template"),
            tests_dir=Path("conf/common/test"),
            hook_dir=Path("conf/common/hook"),
            test_scenario=test_scenario_path,
            output_dir=tmp_path,
            enable_cache_without_check=False,
            single_sbatch=False,
            log_file="debug.log",
        )
        with (
            patch("asyncio.sleep", return_value=None),
            patch("cloudai.systems.slurm.SlurmSystem.is_job_completed", return_value=True),
            patch("cloudai.systems.slurm.SlurmSystem.is_job_running", return_value=True),
            patch("cloudai.util.command_shell.CommandShell.execute") as mock_execute,
        ):
            mock_process = Mock()
            mock_process.poll.return_value = 0
            mock_process.returncode = 0
            mock_process.communicate.return_value = ("", "")
            mock_execute.return_value = mock_process

            handle_dry_run_and_run(args)

        return (tmp_path, scenario)

    def test_the_only_results_dir_created(self, do_dry_run: tuple[Path, dict]) -> None:
        tmp_path = do_dry_run[0]
        results_output_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(results_output_dirs) == 1, "No result directory found or multiple directories found."

    def test_number_of_cases(self, do_dry_run: tuple[Path, dict]) -> None:
        tmp_path, scenario = do_dry_run
        results_output_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        results_output = results_output_dirs[0]

        test_dirs = list(d for d in results_output.iterdir() if d.is_dir())

        if scenario["expected_dirs_number"] is not None:
            assert len(test_dirs) == scenario["expected_dirs_number"], "Dirs number in output is not as expected"

        for td in test_dirs:
            assert td.is_dir(), "Invalid test directory"
            assert "Tests." in td.name, "Invalid test directory name"

    def test_log_file(self, do_dry_run: tuple[Path, dict]) -> None:
        tmp_path, scenario = do_dry_run
        log_file_path = tmp_path / scenario["log_file"]
        assert log_file_path.exists(), f"Log file {log_file_path} was not created"

    def test_details_is_dumped_and_valid(self, do_dry_run: tuple[Path, dict]) -> None:
        tmp_path, scenario = do_dry_run

        num_cases = scenario["expected_dirs_number"]
        details_tomls = list(tmp_path.glob(f"**/{CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME}"))
        assert len(details_tomls) == num_cases, "Details files number is not as expected"

        for details_toml in details_tomls:
            TestRunDetails.model_validate(toml.load(details_toml))


@pytest.fixture
def partial_tr(slurm_system: SlurmSystem) -> partial[TestRun]:
    return partial(TestRun, num_nodes=1, nodes=[], output_path=slurm_system.output_path)


def create_test_run(partial_tr: partial[TestRun], name: str, test_definition: TestDefinition) -> TestRun:
    tr = partial_tr(name=name, test=test_definition)
    return tr


def build_special_test_run(
    partial_tr: partial[TestRun], param: str, test_mapping: Dict[str, Callable[[], TestRun]]
) -> Tuple[TestRun, str, Optional[str]]:
    if "gpt" in param:
        test_type = "gpt"
        tr = create_test_run(
            partial_tr,
            test_type,
            GPTTestDefinition(
                name=test_type,
                description=test_type,
                test_template_name=test_type,
                cmd_args=GPTCmdArgs(
                    fdl_config="fdl/config", docker_image_url="https://docker/url", output_path="/some/output/path"
                ),
                extra_env_vars={"COMBINE_THRESHOLD": "1"},
            ),
        )
    elif "grok" in param:
        test_type = "grok"
        tr = create_test_run(
            partial_tr,
            test_type,
            GrokTestDefinition(
                name=test_type,
                description=test_type,
                test_template_name=test_type,
                cmd_args=GrokCmdArgs(
                    fdl_config="fdl/config", docker_image_url="https://docker/url", output_path="/some/output/path"
                ),
                extra_env_vars={"COMBINE_THRESHOLD": "1"},
            ),
        )
    elif "nemo-run" in param:
        test_type = "nemo-run"
        tr = create_test_run(
            partial_tr,
            test_type,
            NeMoRunTestDefinition(
                name=test_type,
                description=test_type,
                test_template_name=test_type,
                cmd_args=NeMoRunCmdArgs(
                    docker_image_url="nvcr.io/nvidia/nemo:24.09", task="pretrain", recipe_name="llama_3b"
                ),
            ),
        )
    elif "nemo-launcher" in param:
        test_type = "nemo-launcher"
        tr = create_test_run(
            partial_tr,
            test_type,
            NeMoLauncherTestDefinition(
                name="nemo-launcher",
                description="nemo-launcher",
                test_template_name="nemo-launcher",
                cmd_args=NeMoLauncherCmdArgs(),
                extra_env_vars={"VAR": r"$(scontrol show hostname \"${SLURM_STEP_NODELIST}\" | head -n1)"},
            ),
        )
    else:
        raise ValueError(f"Unknown test type: {param}")

    if "pre-test" in param:
        pre_test_tr = test_mapping["nccl"]()
        tr.pre_test = TestScenario(name=f"{pre_test_tr.name} NCCL pre-test", test_runs=[pre_test_tr])
    if test_type in ("nemo-run", "nemo-launcher"):
        return tr, f"{param}.sbatch", None
    return tr, f"{param}.sbatch", f"{test_type}.run"


@pytest.fixture(
    params=[
        "ucc",
        "ddlb",
        "nccl",
        "sleep",
        "gpt-pre-test",
        "gpt-no-hook",
        "grok-pre-test",
        "grok-no-hook",
        "nemo-launcher",
        "nemo-run-pre-test",
        "nemo-run-no-hook",
        "nemo-run-vboost",
        "slurm_container",
        "megatron-run",
        "triton-inference",
        "nixl_bench",
        "ai-dynamo",
        "nixl-perftest",
        "nixl-kvbench",
        "deepep-benchmark",
        "osu-bench",
    ]
)
def test_req(request, slurm_system: SlurmSystem, partial_tr: partial[TestRun]) -> Tuple[TestRun, str, Optional[str]]:
    test_mapping: Dict[str, Callable[[], TestRun]] = {
        "ucc": lambda: create_test_run(
            partial_tr,
            "ucc",
            UCCTestDefinition(
                name="ucc",
                description="ucc",
                test_template_name="ucc",
                cmd_args=UCCCmdArgs(docker_image_url="nvcr.io#nvidia/pytorch:24.02-py3"),
            ),
        ),
        "nccl": lambda: create_test_run(
            partial_tr,
            "nccl",
            NCCLTestDefinition(
                name="nccl",
                description="nccl",
                test_template_name="nccl",
                cmd_args=NCCLCmdArgs(docker_image_url="nvcr.io#nvidia/pytorch:24.02-py3"),
            ),
        ),
        "ddlb": lambda: create_test_run(
            partial_tr,
            "ddlb",
            DDLBTestDefinition(
                name="ddlb",
                description="ddlb",
                test_template_name="ddlb",
                cmd_args=DDLBCmdArgs(
                    docker_image_url="docker/image:url",
                    primitive="tp_columnwise",
                    m=1024,
                    n=128,
                    k=1024,
                    dtype="float16",
                    num_iterations=50,
                    num_warmups=5,
                    impl="pytorch;backend=nccl;order=AG_before",
                ),
            ),
        ),
        "osu-bench": lambda: create_test_run(
            partial_tr,
            "osu-bench",
            OSUBenchTestDefinition(
                name="osu-bench",
                description="osu-bench",
                test_template_name="osu-bench",
                cmd_args=OSUBenchCmdArgs(
                    docker_image_url="nvcr.io#nvidia/pytorch:24.02-py3",
                    benchmarks_dir="/opt/hpcx/ompi/tests/osu-micro-benchmarks",
                    benchmark="osu_allreduce",
                    iterations=10,
                    message_size="1024",
                ),
            ),
        ),
        "sleep": lambda: create_test_run(
            partial_tr,
            "sleep",
            SleepTestDefinition(name="sleep", description="sleep", test_template_name="sleep", cmd_args=SleepCmdArgs()),
        ),
        "slurm_container": lambda: create_test_run(
            partial_tr,
            "slurm_container",
            SlurmContainerTestDefinition(
                name="slurm_container",
                description="slurm_container",
                test_template_name="slurm_container",
                cmd_args=SlurmContainerCmdArgs(docker_image_url="https://docker/url", cmd="pwd ; ls"),
            ),
        ),
        "megatron-run": lambda: create_test_run(
            partial_tr,
            "megatron-run",
            MegatronRunTestDefinition(
                name="megatron-run",
                description="megatron-run",
                test_template_name="megatron-run",
                cmd_args=MegatronRunCmdArgs(
                    docker_image_url="nvcr.io/nvidia/megatron:24.09",
                    run_script=Path.cwd() / "run.py",
                    save=Path.cwd(),
                    load=Path.cwd(),
                    tokenizer_model=Path.cwd() / "model.m",
                ),
                extra_container_mounts=["$PWD"],
            ),
        ),
        "nemo-run": lambda: create_test_run(
            partial_tr,
            "nemo-run",
            NeMoRunTestDefinition(
                name="nemo-run",
                description="Test enabling vboost",
                test_template_name="nemo-run",
                cmd_args=NeMoRunCmdArgs(
                    docker_image_url="nvcr.io/nvidia/nemo:24.09",
                    task="pretrain",
                    recipe_name="llama_3b",
                ),
            ),
        ),
        "triton-inference": lambda: create_test_run(
            partial_tr,
            "triton-inference",
            TritonInferenceTestDefinition(
                name="triton-inference",
                description="triton-inference",
                test_template_name="triton-inference",
                cmd_args=TritonInferenceCmdArgs(
                    server_docker_image_url="nvcr.io/nim/deepseek-ai/deepseek-r1:1.7.2",
                    client_docker_image_url="nvcr.io/nvidia/tritonserver:25.01-py3-sdk",
                    served_model_name="model",
                    tokenizer="tok",
                ),
            ),
        ),
        "nixl_bench": lambda: create_test_run(
            partial_tr,
            "nixl_bench",
            NIXLBenchTestDefinition(
                name="nixl_bench",
                description="nixl_bench",
                test_template_name="nixl_bench",
                cmd_args=NIXLBenchCmdArgs.model_validate(
                    {
                        "docker_image_url": "url.com/docker:2",
                        "path_to_benchmark": "./nixlbench",
                        "backend": "UCX",
                    }
                ),
            ),
        ),
        "nixl-perftest": lambda: create_test_run(
            partial_tr,
            "nixl-perftest",
            NixlPerftestTestDefinition(
                name="nixl-perftest",
                description="nixl-perftest",
                test_template_name="nixl-perftest",
                cmd_args=NixlPerftestCmdArgs(
                    docker_image_url="url.com/docker:tag",
                    subtest="sequential-ct-perftest",
                    etcd_path="etcd",
                    num_user_requests=2,
                    batch_size=1,
                    num_prefill_nodes=1,
                    num_decode_nodes=1,
                    model="model-name",
                ),
            ),
        ),
        "nixl-kvbench": lambda: create_test_run(
            partial_tr,
            "nixl-kvbench",
            NIXLKVBenchTestDefinition(
                name="nixl-perftest",
                description="nixl-perftest",
                test_template_name="nixl-perftest",
                cmd_args=NIXLKVBenchCmdArgs.model_validate(
                    {
                        "docker_image_url": "url.com/docker:tag",
                        "backend": "UCX",
                        "kvbench_script": "path/to/kvbench_script.sh",
                        "python_executable": "path/to/python",
                    }
                ),
            ),
        ),
        "ai-dynamo": lambda: create_test_run(
            partial_tr,
            "ai-dynamo",
            AIDynamoTestDefinition(
                name="ai-dynamo",
                description="AI Dynamo test",
                test_template_name="ai-dynamo",
                dynamo_repo=GitRepo(
                    url="https://github.com/ai-dynamo/dynamo.git",
                    commit="f7e468c7e8ff0d1426db987564e60572167e8464",
                    installed_path=slurm_system.install_path,
                ),
                cmd_args=AIDynamoCmdArgs(
                    docker_image_url="nvcr.io/nvidia/ai-dynamo:24.09",
                    dynamo=AIDynamoArgs(
                        model="model",
                        backend="vllm",
                        workspace_path="/workspace",
                        prefill_worker=PrefillWorkerArgs(
                            **{
                                "num-nodes": 1,
                                "ServiceArgs": {"workers": 1, "resources": {"gpu": "8"}},
                            }
                        ),
                        decode_worker=DecodeWorkerArgs(
                            **{
                                "num-nodes": 1,
                                "ServiceArgs": {"workers": 1, "resources": {"gpu": "8"}},
                            }
                        ),
                    ),
                    genai_perf=GenAIPerfArgs(
                        **{
                            "streaming": True,
                            "extra-inputs": '{"temperature": 0.7, "max_tokens": 128}',
                            "output-tokens-mean": 128,
                            "random-seed": 42,
                            "request-count": 100,
                            "synthetic-input-tokens-mean": 550,
                            "warmup-request-count": 10,
                        }
                    ),
                ),
            ),
        ),
        "deepep-benchmark": lambda: create_test_run(
            partial_tr,
            "deepep-benchmark",
            DeepEPTestDefinition(
                name="deepep-benchmark",
                description="DeepEP MoE Benchmark",
                test_template_name="deepep-benchmark",
                cmd_args=DeepEPCmdArgs(
                    docker_image_url="docker/image:url",
                ),
            ),
        ),
    }

    if request.param.startswith(("gpt-", "grok-", "nemo-run-", "nemo-launcher")):
        tr, sbatch_file, run_script = build_special_test_run(partial_tr, request.param, test_mapping)

        if request.param == "nemo-run-vboost":
            tr.test.extra_env_vars["ENABLE_VBOOST"] = "1"

        return tr, sbatch_file, run_script

    if request.param in test_mapping:
        tr = test_mapping[request.param]()
        if request.param.startswith("triton-inference"):
            tr.num_nodes = 3
            tr.test.extra_env_vars["NIM_MODEL_NAME"] = str(tr.output_path)
            tr.test.extra_env_vars["NIM_CACHE_PATH"] = str(tr.output_path)
        if request.param in {"nixl_bench", "nixl-kvbench"}:
            tr.num_nodes = 2
        if request.param == "ai-dynamo":
            tr.num_nodes = 2
            hf_home = tr.output_path / "hf_home"
            hf_home.mkdir(parents=True, exist_ok=True)
            tr.test.cmd_args.huggingface_home_host_path = str(hf_home)
        if request.param == "deepep-benchmark":
            tr.num_nodes = 2
        return tr, f"{request.param}.sbatch", None

    raise ValueError(f"Unknown test: {request.param}")


def test_sbatch_generation(slurm_system: SlurmSystem, test_req: tuple[TestRun, str]):
    slurm_system.output_path.mkdir(parents=True, exist_ok=True)
    slurm_system.container_mount_home = True
    slurm_system.supports_gpu_directives_cache = True

    tr = test_req[0]

    ref = (Path(__file__).parent / "ref_data" / test_req[1]).read_text().strip()
    ref = (
        ref.replace("__OUTPUT_DIR__", str(slurm_system.output_path.parent))
        .replace("__JOB_NAME__", "job_name")
        .replace("__CLOUDAI_DIR__", str(Path(__file__).parent.parent))
        .replace("__INSTALL_DIR__", str(slurm_system.install_path.absolute()))
    )
    ref = ref.replace("__CLOUDAI_VERSION__", version("cloudai"))

    runner = SlurmRunner(
        mode="run",
        system=slurm_system,
        test_scenario=TestScenario(name="tc", test_runs=[tr]),
        output_path=slurm_system.output_path,
    )
    cmd_gen = runner.get_cmd_gen_strategy(slurm_system, tr)
    if isinstance(cmd_gen, SlurmCommandGenStrategy):
        cmd_gen.job_name = Mock(return_value="job_name")
    if isinstance(cmd_gen, NeMoLauncherSlurmCommandGenStrategy):
        cmd_gen.job_prefix = "test_account-cloudai.nemo"
    sbatch_script = cmd_gen.gen_exec_command().split()[-1]
    if "nemo-launcher" in test_req[1]:
        sbatch_script = slurm_system.output_path / "generated_command.sh"
    curr = Path(sbatch_script).read_text().strip()

    assert curr == ref

    run_script = test_req[-1]
    if run_script:
        curr_run_script = Path(slurm_system.output_path / "run.sh").read_text()
        ref_run_script = (Path(__file__).parent / "ref_data" / run_script).read_text()
        assert curr_run_script == ref_run_script

    if test_req[1] == "triton-inference.sbatch":
        wrapper_file = slurm_system.output_path / "start_server_wrapper.sh"
        assert wrapper_file.exists(), "start_server_wrapper.sh was not generated"
        curr_wrapper = wrapper_file.read_text().strip()
        ref_wrapper = (
            (Path(__file__).parent / "ref_data" / "triton-inference-start_server_wrapper.sh").read_text().strip()
        )
        ref_wrapper = ref_wrapper.replace("__OUTPUT_DIR__", str(slurm_system.output_path.parent))
        assert curr_wrapper == ref_wrapper, "start_server_wrapper.sh does not match reference"
