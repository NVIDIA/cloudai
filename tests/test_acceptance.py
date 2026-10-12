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
import getpass
import json
import shlex
import subprocess
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
from unittest.mock import Mock, patch

import pytest
import toml

import cloudai.cli.cli
import cloudai.models.output
from cloudai.cli import setup_logging
from cloudai.core import METRIC_ERROR, CommandGenStrategy, TestDefinition, TestRun, TestScenario
from cloudai.models.scenario import TestRunDetails
from cloudai.systems.slurm import SingleSbatchRunner, SlurmAPIConfig, SlurmCommandGenStrategy, SlurmRunner, SlurmSystem
from cloudai.workloads.ai_dynamo import (
    AIDynamoArgs,
    AIDynamoCmdArgs,
    AIDynamoTestDefinition,
    AIPerf,
    AIPerfPhase,
    DCGMExporter,
    GenAIPerf,
    WorkerBaseArgs,
    WorkerConfig,
)
from cloudai.workloads.ddlb import DDLBCmdArgs, DDLBTestDefinition
from cloudai.workloads.deepep import (
    DeepEPCmdArgs,
    DeepEPTestDefinition,
)
from cloudai.workloads.fio import FioCmdArgs, FioTestDefinition
from cloudai.workloads.jax_toolbox import (
    GPTCmdArgs,
    GPTTestDefinition,
    GrokCmdArgs,
    GrokTestDefinition,
)
from cloudai.workloads.megatron_bridge import (
    MegatronBridgeReportGenerationStrategy,
    MegatronBridgeSlurmCommandGenStrategy,
    MegatronBridgeTestDefinition,
)
from cloudai.workloads.megatron_run import (
    MegatronRunCmdArgs,
    MegatronRunTestDefinition,
)
from cloudai.workloads.moe_benchmark import MoEBenchmarkCmdArgs, MoEBenchmarkTestDefinition
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition
from cloudai.workloads.nemo_launcher import (
    NeMoLauncherCmdArgs,
    NeMoLauncherSlurmCommandGenStrategy,
    NeMoLauncherTestDefinition,
)
from cloudai.workloads.nemo_run import NeMoRunCmdArgs, NeMoRunTestDefinition
from cloudai.workloads.nixl_bench import NIXLBenchCmdArgs, NIXLBenchTestDefinition
from cloudai.workloads.nixl_ep import NixlEPCmdArgs, NixlEPTestDefinition
from cloudai.workloads.nixl_kvbench import NIXLKVBenchCmdArgs, NIXLKVBenchTestDefinition
from cloudai.workloads.nixl_perftest import NixlPerftestCmdArgs, NixlPerftestTestDefinition
from cloudai.workloads.osu_bench import OSUBenchCmdArgs, OSUBenchTestDefinition
from cloudai.workloads.sglang import SglangArgs, SglangCmdArgs, SglangTestDefinition
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
from cloudai.workloads.vllm import VllmArgs, VllmCmdArgs, VllmRayStartArgs, VllmTestDefinition

SLURM_TEST_SCENARIOS = [
    {
        "path": Path("conf/common/test_scenario/sleep.toml"),
        "expected_dirs_number": 4,
        "log_file": "sleep_debug.log",
        "tests": [
            ("Tests.sleep1", "sleep", "sleep test"),
            ("Tests.sleep5", "sleep", "sleep test"),
            ("Tests.sleep5_2", "sleep", "sleep test"),
            ("Tests.sleep20", "sleep", "sleep test"),
        ],
    },
    {
        "path": Path("conf/common/test_scenario/ucc_test.toml"),
        "expected_dirs_number": 4,
        "log_file": "ucc_test_debug.log",
        "tests": [
            ("Tests.alltoall", "ucc_base_test", "UCC alltoall"),
            ("Tests.allgather", "ucc_base_test", "UCC allgather"),
            ("Tests.allreduce", "ucc_base_test", "UCC allreduce"),
            ("Tests.reduce_scatter", "ucc_base_test", "UCC reduce_scatter"),
        ],
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
            patch("time.sleep", return_value=None),
            patch("cloudai.systems.slurm.SlurmSystem.is_job_completed", return_value=True),
            patch("cloudai.systems.slurm.SlurmSystem.is_job_running", return_value=True),
            patch("cloudai.util.command_shell.CommandShell.execute") as mock_execute,
        ):
            mock_process = Mock()
            mock_process.poll.return_value = 0
            mock_process.returncode = 0
            mock_process.communicate.return_value = ("", "")
            mock_execute.return_value = mock_process

            cloudai.cli.cli.handle_dry_run_and_run(args)

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

    def test_experiment_output_is_dumped_and_valid(self, do_dry_run: tuple[Path, dict]) -> None:
        tmp_path, scenario = do_dry_run
        results_output = next(path for path in tmp_path.iterdir() if path.is_dir())
        experiment = cloudai.models.output.Experiment.model_validate_json(
            (results_output / "experiment.json").read_text()
        )

        assert experiment.model_dump() == {
            "id": results_output.name,
            "name": toml.load(scenario["path"])["name"],
            "system_name": "example-cluster",
            "user": getpass.getuser(),
            "description": None,
            "status": "completed",
            "path": str(results_output.absolute()),
            "start": experiment.start,
            "finish": experiment.finish,
            "duration": experiment.duration,
            "tests": [
                {
                    "id": test_id,
                    "name": name,
                    "description": description,
                    "status": "completed",
                    "path": str(results_output.absolute() / test_id),
                    "metrics": [],
                    "runs": [],
                    "dse": None,
                }
                for test_id, name, description in scenario["tests"]
            ],
        }


@pytest.fixture
def partial_tr(slurm_system: SlurmSystem) -> partial[TestRun]:
    return partial(TestRun, num_nodes=1, nodes=[], output_path=slurm_system.output_path)


def create_test_run(partial_tr: partial[TestRun], name: str, test_definition: TestDefinition, **kwargs) -> TestRun:
    tr = partial_tr(name=name, test=test_definition, **kwargs)
    return tr


def with_installed_dynamo_repo(test_definition: AIDynamoTestDefinition, installed_path: Path) -> AIDynamoTestDefinition:
    """Set the fake installed path used by acceptance command generation."""
    test_definition.repo.installed_path = installed_path
    return test_definition


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
        "nixl_bench-independent",
        "nixl-ep",
        "ai-dynamo",
        "nixl-perftest",
        "nixl-kvbench",
        "moe-benchmark",
        "deepep-benchmark",
        "osu-bench",
        "fio",
        "sglang",
        "sglang-multinode",
        "sglang-disagg",
        "sglang-disagg-2nodes",
        "vllm",
        "vllm-multinode",
        "vllm-disagg",
        "vllm-disagg-2nodes",
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
        "fio": lambda: create_test_run(
            partial_tr,
            "fio",
            FioTestDefinition(
                name="fio",
                description="fio",
                test_template_name="Fio",
                cmd_args=FioCmdArgs(
                    args={
                        "name": "fio-smoke",
                        "filename": "/tmp/cloudai-fio-test",
                        "rw": "randwrite",
                        "bs": "128k",
                        "iodepth": 1,
                        "numjobs": 1,
                        "group_reporting": True,
                        "thread": True,
                    },
                    docker_image_url="openeuler/fio:3.42-oe2403sp3",
                    num_tasks_per_node=2,
                ),
            ),
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
        "nixl-ep": lambda: create_test_run(
            partial_tr,
            "nixl-ep",
            NixlEPTestDefinition(
                name="nixl-ep",
                description="nixl-ep",
                test_template_name="NixlEP",
                cmd_args=NixlEPCmdArgs.model_validate(
                    {
                        "docker_image_url": "docker.io/nvidia/nixl-ep:latest",
                        "elastic_script": "/workspace/nixl/examples/device/ep/tests/elastic/elastic.py",
                        "plan": json.dumps(
                            [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, -6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
                        ),
                        "num_processes_per_node": 4,
                        "service_startup_timeout_seconds": 90,
                        "store_port": 9999,
                        "num_tokens": 256,
                        "num_experts_per_rank": 4,
                        "hidden_dim": 8192,
                        "num_topk": 6,
                        "disable_ll_nvlink": True,
                        "kineto": True,
                    }
                ),
                extra_env_vars={"LD_LIBRARY_PATH": "/workspace/rdma_core/lib:$LD_LIBRARY_PATH"},
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
        "nixl_bench-independent": lambda: create_test_run(
            partial_tr,
            "nixl_bench-independent",
            NIXLBenchTestDefinition(
                name="nixl_bench-independent",
                description="Independent NIXLBench storage processes",
                test_template_name="nixl_bench",
                cmd_args=NIXLBenchCmdArgs.model_validate(
                    {
                        "docker_image_url": "url.com/docker:2",
                        "path_to_benchmark": "./nixlbench",
                        "backend": "POSIX",
                        "launch_mode": "independent",
                        "etcd_endpoints": "",
                    }
                ),
            ),
            num_nodes=4,
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
            with_installed_dynamo_repo(
                AIDynamoTestDefinition(
                    name="ai-dynamo",
                    description="AI Dynamo test",
                    test_template_name="ai-dynamo",
                    cmd_args=AIDynamoCmdArgs(
                        docker_image_url="nvcr.io/nvidia/ai-dynamo:24.09",
                        workloads="aiperf.sh",
                        dynamo=AIDynamoArgs(
                            model="model",
                            backend="vllm",
                            endpoint="v1/chat/completions",
                            workspace_path="/workspace",
                            dcgm_exporter=DCGMExporter(enabled=True, port=9501),
                            prefill_worker=WorkerConfig(
                                cmd="python3 -m dynamo.vllm --is-prefill-worker",
                                worker_initialized_regex="VllmWorker.*has.been.initialized",
                                **{
                                    "num-nodes": 1,
                                    "args": WorkerBaseArgs(),
                                },
                            ),
                            decode_worker=WorkerConfig(
                                cmd="python3 -m dynamo.vllm",
                                worker_initialized_regex="VllmWorker.*has.been.initialized",
                                **{
                                    "num-nodes": 1,
                                    "args": WorkerBaseArgs(),
                                },
                            ),
                        ),
                        genai_perf=GenAIPerf(
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
                        aiperf=AIPerf.model_validate(
                            {
                                "extra-args": "--server-metrics-formats json csv",
                                "args": {
                                    "concurrency": 2,
                                    "request-count": 50,
                                    "synthetic-input-tokens-mean": 300,
                                    "output-tokens-mean": 500,
                                    "server-metrics": "auto",
                                },
                            }
                        ),
                        aiperf_phases=[
                            AIPerfPhase.model_validate({"name": "round_1", "args": {"concurrency": 1}}),
                            AIPerfPhase.model_validate({"name": "round_2", "args": {"request-count": 10}}),
                        ],
                    ),
                ),
                slurm_system.install_path,
            ),
        ),
        "moe-benchmark": lambda: create_test_run(
            partial_tr,
            "moe-benchmark",
            MoEBenchmarkTestDefinition(
                name="moe-benchmark",
                description="MoE Benchmark",
                test_template_name="MoEBenchmark",
                cmd_args=MoEBenchmarkCmdArgs(
                    docker_image_url="docker/image:url",
                    deepep_versions=["legacy", "elastic", "nixl_ep", "uccl_ep", "nccl_ep", "deepep_hybrid"],
                ),
            ),
        ),
        "deepep-benchmark": lambda: create_test_run(
            partial_tr,
            "deepep-benchmark",
            DeepEPTestDefinition(
                name="deepep-benchmark",
                description="DeepEP internode test",
                test_template_name="DeepEP",
                cmd_args=DeepEPCmdArgs(
                    docker_image_url="docker/image:url",
                    subtest_name="test_internode",
                ),
            ),
        ),
        "vllm": lambda: create_test_run(
            partial_tr,
            "vllm",
            VllmTestDefinition(
                name="vllm",
                description="vLLM benchmark",
                test_template_name="Vllm",
                cmd_args=VllmCmdArgs(
                    docker_image_url="nvcr.io/nvidia/vllm:latest",
                    model="Qwen/Qwen3-0.6B",
                ),
                extra_env_vars={"CUDA_VISIBLE_DEVICES": "0"},
            ),
        ),
        "vllm-multinode": lambda: create_test_run(
            partial_tr,
            "vllm-multinode",
            VllmTestDefinition(
                name="vllm-multinode",
                description="vLLM distributed benchmark on 2 nodes",
                test_template_name="Vllm",
                cmd_args=VllmCmdArgs(
                    docker_image_url="nvcr.io/nvidia/vllm:latest",
                    model="Qwen/Qwen3-0.6B",
                    decode=VllmArgs.model_validate(
                        {
                            "tensor_parallel_size": 8,
                            "ray_head": VllmRayStartArgs.model_validate(
                                {"num_gpus": 4, "num_cpus": 64, "disable_usage_stats": True}
                            ),
                            "ray_worker": VllmRayStartArgs.model_validate(
                                {"num_gpus": 4, "num_cpus": 64, "disable_usage_stats": True}
                            ),
                        }
                    ),
                ),
                extra_env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            ),
        ),
        "sglang": lambda: create_test_run(
            partial_tr,
            "sglang",
            SglangTestDefinition(
                name="sglang",
                description="SGLang benchmark",
                test_template_name="sglang",
                cmd_args=SglangCmdArgs(
                    docker_image_url="docker.io/lmsysorg/sglang:dev",
                    model="Qwen/Qwen3-8B",
                ),
                extra_env_vars={"CUDA_VISIBLE_DEVICES": "0"},
            ),
        ),
        "sglang-multinode": lambda: create_test_run(
            partial_tr,
            "sglang-multinode",
            SglangTestDefinition(
                name="sglang-multinode",
                description="SGLang distributed benchmark on 2 nodes",
                test_template_name="sglang",
                cmd_args=SglangCmdArgs(
                    docker_image_url="docker.io/lmsysorg/sglang:dev",
                    model="Qwen/Qwen3-8B",
                    decode=SglangArgs.model_validate({"tp": 8}),
                ),
                extra_env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            ),
        ),
        "sglang-disagg": lambda: create_test_run(
            partial_tr,
            "sglang-disagg",
            SglangTestDefinition(
                name="sglang-disagg",
                description="SGLang disaggregated benchmark",
                test_template_name="sglang",
                cmd_args=SglangCmdArgs(
                    docker_image_url="docker.io/lmsysorg/sglang:dev",
                    model="Qwen/Qwen3-8B",
                    prefill=SglangArgs(),
                ),
                extra_env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            ),
        ),
        "sglang-disagg-2nodes": lambda: create_test_run(
            partial_tr,
            "sglang-disagg-2nodes",
            SglangTestDefinition(
                name="sglang-disagg-2nodes",
                description="SGLang disaggregated benchmark on 2 nodes",
                test_template_name="sglang",
                cmd_args=SglangCmdArgs(
                    docker_image_url="docker.io/lmsysorg/sglang:dev",
                    model="Qwen/Qwen3-8B",
                    prefill=SglangArgs(),
                ),
                extra_env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            ),
        ),
        "vllm-disagg": lambda: create_test_run(
            partial_tr,
            "vllm-disagg",
            VllmTestDefinition(
                name="vllm-disagg",
                description="vLLM disaggregated benchmark",
                test_template_name="Vllm",
                cmd_args=VllmCmdArgs(
                    docker_image_url="nvcr.io/nvidia/vllm:latest",
                    model="Qwen/Qwen3-0.6B",
                    prefill=VllmArgs(),
                ),
                extra_env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            ),
        ),
        "vllm-disagg-2nodes": lambda: create_test_run(
            partial_tr,
            "vllm-disagg-2nodes",
            VllmTestDefinition(
                name="vllm-disagg-2nodes",
                description="vLLM disaggregated benchmark on 2 nodes",
                test_template_name="Vllm",
                cmd_args=VllmCmdArgs(
                    docker_image_url="nvcr.io/nvidia/vllm:latest",
                    model="Qwen/Qwen3-0.6B",
                    prefill=VllmArgs(),
                ),
                extra_env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
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
        if request.param == "nixl-ep":
            tr.num_nodes = 3
        if request.param == "ai-dynamo":
            tr.num_nodes = 2
        if request.param in {"moe-benchmark", "deepep-benchmark", "fio"}:
            tr.num_nodes = 2
        if request.param in {"sglang-multinode", "sglang-disagg-2nodes", "vllm-multinode", "vllm-disagg-2nodes"}:
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

    if test_req[1] == "nixl-ep.sbatch":
        launcher_file = slurm_system.output_path / "nixl-ep-launch.sh"
        assert launcher_file.exists(), "nixl-ep-launch.sh was not generated"
        curr_launcher = launcher_file.read_text().strip()
        ref_launcher = (Path(__file__).parent / "ref_data" / "nixl-ep-launch.sh").read_text().strip()
        ref_launcher = ref_launcher.replace("__OUTPUT_DIR__", str(slurm_system.output_path.parent)).replace(
            "__INSTALL_DIR__", str(slurm_system.install_path.absolute())
        )
        assert curr_launcher == ref_launcher, "nixl-ep-launch.sh does not match reference"

    if test_req[1] == "nixl_bench-independent.sbatch":
        task_script = slurm_system.output_path / "nixlbench.sh"
        ref_script = (Path(__file__).parent / "ref_data" / "nixl_bench-independent.sh").read_text().strip()
        ref_script = ref_script.replace("__OUTPUT_DIR__", str(slurm_system.output_path.parent))
        assert task_script.read_text().strip() == ref_script

    if test_req[1] == "ai-dynamo.sbatch":
        aiperf_script = slurm_system.output_path / "aiperf.sh"
        assert aiperf_script.exists(), "aiperf.sh was not generated"
        curr_aiperf = aiperf_script.read_text().strip()
        ref_aiperf = (Path(__file__).parent / "ref_data" / "ai-dynamo-aiperf.sh").read_text().strip()
        assert curr_aiperf == ref_aiperf, "aiperf.sh does not match reference"


class TestMBridgeAcceptance:
    """Exercise the runner lifecycle; only upstream generation and remote execution are simulated."""

    @pytest.fixture
    def mbridge_tr(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestRun:
        monkeypatch.setenv("HF_TOKEN", "dummy_token")
        config = toml.load("conf/experimental/megatron_bridge/test/h100/megatron_bridge_qwen_30b.toml")
        config["cmd_args"]["executor_env_vars"] = {"PRIORITY": "executor", "EXECUTOR_ONLY": "executor"}
        config["extra_env_vars"] = {"PRIORITY": "test", "TASK_RANK": "$SLURM_PROCID"}
        tdef = MegatronBridgeTestDefinition.model_validate(config)
        tdef.megatron_bridge_repo.installed_path = tmp_path / "bridge"
        tdef.python_executable.venv_path = tmp_path / "venv"
        tdef.docker_image.installed_path = tmp_path / "nemo.sqsh"
        return TestRun(
            name="mbridge",
            test=tdef,
            num_nodes=2,
            nodes=[],
            output_path=tmp_path / "job",
            time_limit="00:10:00",
            extra_srun_args="--cpu-bind=none",
        )

    @pytest.mark.parametrize("transport", ["cli", "rest", "single-sbatch"])
    @pytest.mark.parametrize("vboost", [True, False])
    def test_submission_and_reporting(
        self,
        slurm_system: SlurmSystem,
        mbridge_tr: TestRun,
        transport: str,
        vboost: bool,
    ) -> None:
        tr = mbridge_tr
        tdef = tr.test
        assert isinstance(tdef, MegatronBridgeTestDefinition)
        tdef.cmd_args.enable_vboost = vboost
        tdef.cmd_args.enable_nsys = True
        tdef.cmd_args.profiling_ranks = "0,8"
        slurm_system.global_env_vars = {"PRIORITY": "global", "GLOBAL_ONLY": "global", "ENABLE_VBOOST": "1"}
        slurm_system.extra_sbatch_args = ["--open-mode=truncate"]
        slurm_system.extra_srun_args = "--kill-on-bad-exit=1"
        if transport == "rest":
            slurm_system.slurm_api = SlurmAPIConfig(url="https://slurm.example.com")
        if transport != "single-sbatch":
            for hook in ("pre_test", "post_test"):
                hook_tr = TestRun(
                    name=hook,
                    num_nodes=2,
                    nodes=[],
                    test=NCCLTestDefinition(
                        name=hook,
                        description=hook,
                        test_template_name="NcclTest",
                        cmd_args=NCCLCmdArgs(docker_image_url="nccl.sqsh"),
                    ),
                )
                setattr(tr, hook, TestScenario(name=hook, test_runs=[hook_tr]))
        runner_cls = SingleSbatchRunner if transport == "single-sbatch" else SlurmRunner
        runner = runner_cls(
            mode="run",
            system=slurm_system,
            test_scenario=TestScenario(name="mbridge-acceptance", test_runs=[tr]),
            output_path=slurm_system.output_path,
        )
        if isinstance(runner, SingleSbatchRunner):
            tr = next(runner.all_trs)

        def generate_nemo_script(command: str, **kwargs: object) -> subprocess.CompletedProcess[str]:
            args = shlex.split(command)
            assert "-d" in args
            assert args[args.index("-vb") + 1] == "false"
            assert "-en" in args and args[args.index("--profiling_ranks") + 1] == "0,8"
            assert args[args.index("-cb") + 1] == "source /cloudai_run_results/env_vars.sh"
            source = Path("tests/ref_data/megatron-bridge-nemo.sbatch").read_text()
            source = source.replace("__OUTPUT__", str(tr.output_path)).replace(
                "__IMAGE__", str(tdef.docker_image.installed_path)
            )
            source = source.replace("__PROFILE__", "nsys profile -t cuda,nvtx")
            script = tr.output_path / "experiments/run/training_sbatch.sh"
            script.parent.mkdir(parents=True)
            script.write_text(source)
            return subprocess.CompletedProcess(command, 0)

        response = Mock()
        response.json.return_value = {"job_id": 123, "errors": []}
        process = Mock()
        process.communicate.return_value = ("Submitted batch job 123", "")
        process.returncode = 0
        with (
            patch(
                "cloudai.workloads.megatron_bridge.slurm_command_gen_strategy.subprocess.run",
                side_effect=generate_nemo_script,
            ) as launch,
            patch("cloudai.util.command_shell.CommandShell.execute", return_value=process) as cli,
            patch("cloudai.systems.slurm.slurm_rest_client.requests.request", return_value=response) as rest,
        ):
            if isinstance(runner, SingleSbatchRunner):
                content = runner.gen_sbatch_content()
                assert ("vboost.out" in content) is vboost
            else:
                runner.submit_test(tr)
                assert runner.jobs[0].id == 123
                content = (tr.output_path / "cloudai_sbatch_script.sh").read_text()
                assert content.index("/pre_test/") < content.index("python train.py") < content.index("/post_test/")
                assert "exit $?\n)" in content
                assert "#SBATCH -N 2" in content
                assert "#SBATCH --exclusive" in content and "#SBATCH --mem=0" in content
                assert "upstream-node" not in content and "upstream-partition" not in content
                assert "#SBATCH --open-mode=append" not in content
                assert ("boost-slider --vboost 1" in content) is vboost
                if transport == "rest":
                    cli.assert_not_called()
                    payload = rest.call_args.kwargs["json"]
                    assert payload["script"] == content
                    assert payload["job"]["nodes"] == [2, 2]
                    assert payload["job"]["exclusive"] == "exclusive"
                    assert payload["job"]["memory_per_node"] == 0
                    assert payload["job"]["open_mode"] == "truncate"
                else:
                    rest.assert_not_called()
                    cli.assert_called_once_with(f"sbatch {tr.output_path / 'cloudai_sbatch_script.sh'}")
            launch.assert_called_once()
            # A fresh strategy, as used for metadata/reporting, must reuse the retained NeMo artifacts.
            cmd_gen = MegatronBridgeSlurmCommandGenStrategy(slurm_system, tr)
            cmd_gen.generate_test_command()
            launch.assert_called_once()

        subprocess.run(["bash", "-n"], input=content, text=True, check=True)
        assert "slurm-metadata.sh" in content and "nsys profile -t cuda,nvtx" in content
        assert 'head_node_ip="$head_node"' in content and "hostname --ip-address" not in content
        assert "export PRIORITY=upstream" not in content and "export UPSTREAM_ONLY=retained" in content
        assert "--kill-on-bad-exit=1 --cpu-bind=none" in content
        env = (tr.output_path / "env_vars.sh").read_text()
        assert 'export PRIORITY="test"' in env
        assert 'export EXECUTOR_ONLY="executor"' in env and 'export GLOBAL_ONLY="global"' in env
        assert 'export TASK_RANK="$SLURM_PROCID"' in env
        assert f'export ENABLE_VBOOST="{int(vboost)}"' in env
        details = toml.load(tr.output_path / "test-run.toml")
        assert details["test_definition"]["cmd_args"]["hf_token"] == "<REDACTED>"
        assert "dummy_token" not in (tr.output_path / "test-run.toml").read_text()
        assert tdef.cmd_args.hf_token == "dummy_token"

        training_log = tr.output_path / "experiments/run/log_123.out"
        training_log.write_text(
            "Step Time : 2.0s GPU utilization: 100.0MODEL_TFLOP/s/GPU\n"
            "Step Time : 4.0s GPU utilization: 200.0TFLOP/s/GPU\n"
        )
        report = MegatronBridgeReportGenerationStrategy(slurm_system, tr)
        assert tdef.was_run_successful(tr).is_successful
        assert report.get_metric("step-time") == 3.0 and report.get_metric("tflops-per-gpu") == 150.0
        report.generate_report()
        assert str(training_log) in (tr.output_path / "report.txt").read_text()
        cmd_gen.cleanup_job_artifacts()
        assert "dummy_token" not in (tr.output_path / "cloudai_megatron_bridge_launcher.sh").read_text()

    @pytest.mark.parametrize("returncode", [0, 1])
    def test_generation_failure_does_not_submit(
        self,
        slurm_system: SlurmSystem,
        mbridge_tr: TestRun,
        returncode: int,
    ) -> None:
        runner = SlurmRunner(
            mode="run",
            system=slurm_system,
            test_scenario=TestScenario(name="mbridge-failure", test_runs=[mbridge_tr]),
            output_path=slurm_system.output_path,
        )
        with (
            patch(
                "cloudai.workloads.megatron_bridge.slurm_command_gen_strategy.subprocess.run",
                return_value=subprocess.CompletedProcess("", returncode),
            ),
            patch.object(SlurmSystem, "submit_job") as submit,
            pytest.raises(RuntimeError, match=r"Megatron-Bridge dry-run.*See .*launcher.log"),
        ):
            runner.submit_test(mbridge_tr)
        submit.assert_not_called()

    @pytest.mark.parametrize("step_times", [None, [], [2.0], list(range(1, 13))])
    def test_legacy_reporting(
        self,
        slurm_system: SlurmSystem,
        mbridge_tr: TestRun,
        step_times: list[float] | None,
    ) -> None:
        tr = mbridge_tr
        tr.output_path.mkdir()
        log = tr.output_path / "cloudai_megatron_bridge_launcher.log"
        if step_times is not None:
            log.write_text(
                "\n".join(f"Step Time : {step}s GPU utilization: 100MODEL_TFLOP/s/GPU" for step in step_times)
            )
        report = MegatronBridgeReportGenerationStrategy(slurm_system, tr)
        assert report.can_handle_directory() is (step_times is not None)
        assert report.results_file == log
        assert tr.test.was_run_successful(tr).is_successful is bool(step_times)
        assert report.get_metric("unsupported") == METRIC_ERROR
        report.generate_report()
        if step_times:
            last_steps = step_times[-10:]
            assert report.get_metric("default") == sum(last_steps) / len(last_steps)
            assert report.get_metric("tflops-per-gpu") == 100
            assert "TFLOP/s per GPU" in (tr.output_path / "report.txt").read_text()
        else:
            assert report.get_metric("default") == METRIC_ERROR
            assert (tr.output_path / "report.txt").exists() is (step_times is not None)
