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
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Type
from unittest.mock import Mock, patch

import pytest

from cloudai import CommandGenStrategy, Test, TestDefinition, TestRun, TestScenario, TestTemplate
from cloudai.cli import handle_dry_run_and_run, setup_logging
from cloudai.systems import SlurmSystem
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
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
from cloudai.workloads.sleep import SleepCmdArgs, SleepSlurmCommandGenStrategy, SleepTestDefinition
from cloudai.workloads.slurm_container import (
    SlurmContainerCmdArgs,
    SlurmContainerCommandGenStrategy,
    SlurmContainerTestDefinition,
)
from cloudai.workloads.slurm_ray_container import (
    SlurmRayContainerCmdArgs,
    SlurmRayContainerCommandGenStrategy,
    SlurmRayContainerTestDefinition,
)
from cloudai.workloads.ucc_test import UCCCmdArgs, UCCTestDefinition, UCCTestSlurmCommandGenStrategy

SLURM_TEST_SCENARIOS = [
    {"path": Path("conf/common/test_scenario/sleep.toml"), "expected_dirs_number": 4, "log_file": "sleep_debug.log"},
    {
        "path": Path("conf/common/test_scenario/ucc_test.toml"),
        "expected_dirs_number": 5,
        "log_file": "ucc_test_debug.log",
    },
]


@pytest.mark.parametrize("scenario", SLURM_TEST_SCENARIOS, ids=lambda x: str(x))
def test_slurm(tmp_path: Path, scenario: Dict):
    test_scenario_path = scenario["path"]
    expected_dirs_number = scenario.get("expected_dirs_number")
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
    )
    with (
        patch("asyncio.sleep", return_value=None),
        patch("cloudai.systems.slurm.SlurmSystem.is_job_completed", return_value=True),
        patch("cloudai.systems.slurm.SlurmSystem.is_job_running", return_value=True),
    ):
        handle_dry_run_and_run(args)

    # Find the directory that was created for the test results
    results_output_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]

    # Assuming there's only one result directory created
    assert len(results_output_dirs) == 1, "No result directory found or multiple directories found."
    results_output = results_output_dirs[0]

    test_dirs = list(results_output.iterdir())

    if expected_dirs_number is not None:
        assert len(test_dirs) == expected_dirs_number, "Dirs number in output is not as expected"

    for td in test_dirs:
        assert td.is_dir(), "Invalid test directory"
        assert "Tests." in td.name, "Invalid test directory name"

    assert log_file_path.exists(), f"Log file {log_file_path} was not created"


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
        test=Test(test_definition=test_definition, test_template=TestTemplate(slurm_system, name=name)),
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
                cmd_args=GPTCmdArgs(fdl_config="fdl/config", docker_image_url="https://docker/url"),
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
                cmd_args=GrokCmdArgs(fdl_config="fdl/config", docker_image_url="https://docker/url"),
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
        "slurm_container",
        "megatron-run",
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
        "slurm_ray_container": lambda: create_test_run(
            partial_tr,
            slurm_system,
            "slurm_ray_container",
            SlurmRayContainerTestDefinition(
                name="slurm_ray_container",
                description="slurm_ray_container",
                test_template_name="slurm_ray_container",
                cmd_args=SlurmRayContainerCmdArgs(
                    docker_image_url="https://docker/url", cmd="pwd ; ls", conda_env="test"
                ),
            ),
            SlurmRayContainerCommandGenStrategy,
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
                    save=Path.cwd() / "save",
                    load=Path.cwd() / "load",
                    tokenizer_model=Path.cwd() / "model.m",
                ),
            ),
            MegatronRunSlurmCommandGenStrategy,
        ),
    }

    if request.param.startswith(("gpt-", "grok-", "nemo-run-", "nemo-launcher")):
        return build_special_test_run(partial_tr, slurm_system, request.param, test_mapping)
    if request.param in test_mapping:
        tr = test_mapping[request.param]()
        return tr, f"{request.param}.sbatch", None
    raise ValueError(f"Unknown test: {request.param}")


def test_sbatch_generation(slurm_system: SlurmSystem, test_req: tuple[TestRun, str]):
    slurm_system.output_path.mkdir(parents=True, exist_ok=True)

    tr = test_req[0]

    ref = (Path(__file__).parent / "ref_data" / test_req[1]).read_text().strip()
    ref = (
        ref.replace("__OUTPUT_DIR__", str(slurm_system.output_path.parent))
        .replace("__JOB_NAME__", "job_name")
        .replace("__CLOUDAI_DIR__", str(Path(__file__).parent.parent))
    )

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
