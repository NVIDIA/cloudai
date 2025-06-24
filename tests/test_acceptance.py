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

import argparse
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Type
from unittest.mock import Mock, patch

import pytest
import toml

from cloudai.cli import handle_dry_run_and_run, setup_logging
from cloudai.core import CommandGenStrategy, Test, TestDefinition, TestRun, TestScenario, TestTemplate
from cloudai.models.scenario import TestRunDetails
from cloudai.systems.slurm import SlurmCommandGenStrategy, SlurmSystem
from cloudai.workloads.ai_dynamo import (
    AIDynamoArgs,
    AIDynamoCmdArgs,
    AIDynamoSlurmCommandGenStrategy,
    AIDynamoTestDefinition,
    FrontendArgs,
    GenAIPerfArgs,
    PrefillWorkerArgs,
    ProcessorArgs,
    RouterArgs,
    VllmWorkerArgs,
)
from cloudai.workloads.jax_toolbox import (
    GPTCmdArgs,
    GPTTestDefinition,
    GrokCmdArgs,
    GrokTestDefinition,
    JaxToolboxSlurmCommandGenStrategy,
)
from cloudai.workloads.megatron_run import (
    MegatronRunCmdArgs,
    MegatronRunSlurmCommandGenStrategy,
    MegatronRunTestDefinition,
)
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition, NcclTestSlurmCommandGenStrategy
from cloudai.workloads.nemo_launcher import (
    NeMoLauncherCmdArgs,
    NeMoLauncherSlurmCommandGenStrategy,
    NeMoLauncherTestDefinition,
)
from cloudai.workloads.nemo_run import NeMoRunCmdArgs, NeMoRunSlurmCommandGenStrategy, NeMoRunTestDefinition
from cloudai.workloads.nixl_bench import NIXLBenchCmdArgs, NIXLBenchSlurmCommandGenStrategy, NIXLBenchTestDefinition
from cloudai.workloads.sleep import SleepCmdArgs, SleepSlurmCommandGenStrategy, SleepTestDefinition
from cloudai.workloads.slurm_container import (
    SlurmContainerCmdArgs,
    SlurmContainerCommandGenStrategy,
    SlurmContainerTestDefinition,
)
from cloudai.workloads.triton_inference import (
    TritonInferenceCmdArgs,
    TritonInferenceSlurmCommandGenStrategy,
    TritonInferenceTestDefinition,
)
from cloudai.workloads.ucc_test import UCCCmdArgs, UCCTestDefinition, UCCTestSlurmCommandGenStrategy

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

        test_dirs = list(results_output.iterdir())

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


def create_test_run(
    partial_tr: partial[TestRun],
    slurm_system: SlurmSystem,
    name: str,
    test_definition: TestDefinition,
    command_gen_strategy: Type[CommandGenStrategy],
) -> TestRun:
    tr = partial_tr(
        name=name,
        test=Test(test_definition=test_definition, test_template=TestTemplate(slurm_system)),
    )
    tr.test.test_template.command_gen_strategy = command_gen_strategy(
        slurm_system, tr.test.test_definition.cmd_args_dict
    )
    if isinstance(tr.test.test_template.command_gen_strategy, SlurmCommandGenStrategy):
        tr.test.test_template.command_gen_strategy.job_name = Mock(return_value="job_name")
    return tr


def build_special_test_run(
    partial_tr: partial[TestRun],
    slurm_system: SlurmSystem,
    param: str,
    test_mapping: Dict[str, Callable[[], TestRun]],
) -> Tuple[TestRun, str, Optional[str]]:
    if "gpt" in param:
        test_type = "gpt"
        tr = create_test_run(
            partial_tr,
            slurm_system,
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
            JaxToolboxSlurmCommandGenStrategy,
        )
    elif "grok" in param:
        test_type = "grok"
        tr = create_test_run(
            partial_tr,
            slurm_system,
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
            JaxToolboxSlurmCommandGenStrategy,
        )
    elif "nemo-run" in param:
        test_type = "nemo-run"
        tr = create_test_run(
            partial_tr,
            slurm_system,
            test_type,
            NeMoRunTestDefinition(
                name=test_type,
                description=test_type,
                test_template_name=test_type,
                cmd_args=NeMoRunCmdArgs(
                    docker_image_url="nvcr.io/nvidia/nemo:24.09", task="pretrain", recipe_name="llama_3b"
                ),
            ),
            NeMoRunSlurmCommandGenStrategy,
        )
    elif "nemo-launcher" in param:
        test_type = "nemo-launcher"
        tr = create_test_run(
            partial_tr,
            slurm_system,
            test_type,
            NeMoLauncherTestDefinition(
                name="nemo-launcher",
                description="nemo-launcher",
                test_template_name="nemo-launcher",
                cmd_args=NeMoLauncherCmdArgs(),
                extra_env_vars={"VAR": r"$(scontrol show hostname \"${SLURM_STEP_NODELIST}\" | head -n1)"},
            ),
            NeMoLauncherSlurmCommandGenStrategy,
        )
        assert isinstance(tr.test.test_template.command_gen_strategy, NeMoLauncherSlurmCommandGenStrategy)
        tr.test.test_template.command_gen_strategy.job_prefix = "test_account-cloudai.nemo"
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
    ]
)
def test_req(request, slurm_system: SlurmSystem, partial_tr: partial[TestRun]) -> Tuple[TestRun, str, Optional[str]]:
    test_mapping: Dict[str, Callable[[], TestRun]] = {
        "ucc": lambda: create_test_run(
            partial_tr,
            slurm_system,
            "ucc",
            UCCTestDefinition(name="ucc", description="ucc", test_template_name="ucc", cmd_args=UCCCmdArgs()),
            UCCTestSlurmCommandGenStrategy,
        ),
        "nccl": lambda: create_test_run(
            partial_tr,
            slurm_system,
            "nccl",
            NCCLTestDefinition(name="nccl", description="nccl", test_template_name="nccl", cmd_args=NCCLCmdArgs()),
            NcclTestSlurmCommandGenStrategy,
        ),
        "sleep": lambda: create_test_run(
            partial_tr,
            slurm_system,
            "sleep",
            SleepTestDefinition(name="sleep", description="sleep", test_template_name="sleep", cmd_args=SleepCmdArgs()),
            SleepSlurmCommandGenStrategy,
        ),
        "slurm_container": lambda: create_test_run(
            partial_tr,
            slurm_system,
            "slurm_container",
            SlurmContainerTestDefinition(
                name="slurm_container",
                description="slurm_container",
                test_template_name="slurm_container",
                cmd_args=SlurmContainerCmdArgs(docker_image_url="https://docker/url", cmd="pwd ; ls"),
            ),
            SlurmContainerCommandGenStrategy,
        ),
        "megatron-run": lambda: create_test_run(
            partial_tr,
            slurm_system,
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
            MegatronRunSlurmCommandGenStrategy,
        ),
        "nemo-run": lambda: create_test_run(
            partial_tr,
            slurm_system,
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
            NeMoRunSlurmCommandGenStrategy,
        ),
        "triton-inference": lambda: create_test_run(
            partial_tr,
            slurm_system,
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
            TritonInferenceSlurmCommandGenStrategy,
        ),
        "nixl_bench": lambda: create_test_run(
            partial_tr,
            slurm_system,
            "nixl_bench",
            NIXLBenchTestDefinition(
                name="nixl_bench",
                description="nixl_bench",
                test_template_name="nixl_bench",
                etcd_image_url="url.com/docker:1",
                cmd_args=NIXLBenchCmdArgs(
                    docker_image_url="url.com/docker:2",
                    etcd_endpoint="http://$SLURM_JOB_MASTER_NODE:2379",
                    path_to_benchmark="./nixlbench",
                ),
            ),
            NIXLBenchSlurmCommandGenStrategy,
        ),
        "ai-dynamo": lambda: create_test_run(
            partial_tr,
            slurm_system,
            "ai-dynamo",
            AIDynamoTestDefinition(
                name="ai-dynamo",
                description="AI Dynamo test",
                test_template_name="ai-dynamo",
                cmd_args=AIDynamoCmdArgs(
                    docker_image_url="nvcr.io/nvidia/ai-dynamo:24.09",
                    served_model_name="llama2-7b",
                    dynamo=AIDynamoArgs(
                        frontend=FrontendArgs(),
                        processor=ProcessorArgs(**{"block-size": 64, "max-model-len": 8192, "router": "kv"}),
                        router=RouterArgs(**{"min-workers": 1}),
                        prefill_worker=PrefillWorkerArgs(
                            **{
                                "num_nodes": 1,
                                "ServiceArgs": {"workers": 1, "resources": {"gpu": "8"}},
                            }
                        ),
                        vllm_worker=VllmWorkerArgs(
                            **{
                                "num_nodes": 1,
                                "ServiceArgs": {"workers": 1, "resources": {"gpu": "8"}},
                            }
                        ),
                    ),
                    genai_perf=GenAIPerfArgs(
                        streaming=True,
                        extra_inputs='{"temperature": 0.7, "max_tokens": 128}',
                        output_tokens_mean=128,
                        random_seed=42,
                        request_count=100,
                        synthetic_input_tokens_mean=550,
                        warmup_request_count=10,
                    ),
                ),
            ),
            AIDynamoSlurmCommandGenStrategy,
        ),
    }

    if request.param.startswith(("gpt-", "grok-", "nemo-run-", "nemo-launcher")):
        tr, sbatch_file, run_script = build_special_test_run(partial_tr, slurm_system, request.param, test_mapping)

        if request.param == "nemo-run-vboost":
            tr.test.extra_env_vars["ENABLE_VBOOST"] = "1"

        return tr, sbatch_file, run_script

    if request.param in test_mapping:
        tr = test_mapping[request.param]()
        if request.param.startswith("triton-inference"):
            tr.num_nodes = 3
            tr.test.test_definition.extra_env_vars["NIM_MODEL_NAME"] = str(tr.output_path)
            tr.test.test_definition.extra_env_vars["NIM_CACHE_PATH"] = str(tr.output_path)
        if request.param == "nixl_bench":
            tr.num_nodes = 2
        if request.param == "ai-dynamo":
            tr.num_nodes = 3
            hf_home = tr.output_path / "hf_home"
            hf_home.mkdir(parents=True, exist_ok=True)
            tr.test.test_definition.extra_env_vars["HF_HOME"] = str(hf_home)
        return tr, f"{request.param}.sbatch", None

    raise ValueError(f"Unknown test: {request.param}")


def test_sbatch_generation(slurm_system: SlurmSystem, test_req: tuple[TestRun, str]):
    slurm_system.output_path.mkdir(parents=True, exist_ok=True)
    slurm_system.container_mount_home = True

    tr = test_req[0]

    ref = (Path(__file__).parent / "ref_data" / test_req[1]).read_text().strip()
    ref = (
        ref.replace("__OUTPUT_DIR__", str(slurm_system.output_path.parent))
        .replace("__JOB_NAME__", "job_name")
        .replace("__CLOUDAI_DIR__", str(Path(__file__).parent.parent))
        .replace("__INSTALL_DIR__", str(slurm_system.install_path.absolute()))
    )
    ref = ref.replace("__CLOUDAI_VERSION__", version("cloudai"))

    sbatch_script = tr.test.test_template.gen_exec_command(tr).split()[-1]
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
