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

import argparse
from functools import partial
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import Mock, patch

import pytest

from cloudai import NcclTest, Test, TestRun, TestScenario, UCCTest
from cloudai.cli import handle_dry_run_and_run, setup_logging
from cloudai.schema.test_template.jax_toolbox.slurm_command_gen_strategy import JaxToolboxSlurmCommandGenStrategy
from cloudai.schema.test_template.jax_toolbox.template import JaxToolbox
from cloudai.schema.test_template.nccl_test.slurm_command_gen_strategy import NcclTestSlurmCommandGenStrategy
from cloudai.schema.test_template.nemo_launcher.slurm_command_gen_strategy import NeMoLauncherSlurmCommandGenStrategy
from cloudai.schema.test_template.nemo_launcher.template import NeMoLauncher
from cloudai.schema.test_template.sleep.slurm_command_gen_strategy import SleepSlurmCommandGenStrategy
from cloudai.schema.test_template.sleep.template import Sleep
from cloudai.schema.test_template.ucc_test.slurm_command_gen_strategy import UCCTestSlurmCommandGenStrategy
from cloudai.systems import SlurmSystem
from cloudai.test_definitions.gpt import GPTCmdArgs, GPTTestDefinition
from cloudai.test_definitions.grok import GrokCmdArgs, GrokTestDefinition
from cloudai.test_definitions.nccl import NCCLCmdArgs, NCCLTestDefinition
from cloudai.test_definitions.nemo_launcher import NeMoLauncherCmdArgs, NeMoLauncherTestDefinition
from cloudai.test_definitions.sleep import SleepCmdArgs, SleepTestDefinition
from cloudai.test_definitions.ucc import UCCCmdArgs, UCCTestDefinition

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
    )
    with patch("asyncio.sleep", return_value=None):
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


@pytest.fixture(
    params=["ucc", "nccl", "sleep", "gpt-pre-test", "gpt-no-hook", "grok-pre-test", "grok-no-hook", "nemo-launcher"]
)
def test_req(request, slurm_system: SlurmSystem, partial_tr: partial[TestRun]) -> tuple[TestRun, str, Optional[str]]:
    if request.param == "ucc":
        tr = partial_tr(
            name="ucc",
            test=Test(
                test_definition=UCCTestDefinition(
                    name="ucc", description="ucc", test_template_name="ucc", cmd_args=UCCCmdArgs()
                ),
                test_template=UCCTest(slurm_system, name="ucc"),
            ),
        )
        tr.test.test_template.command_gen_strategy = UCCTestSlurmCommandGenStrategy(
            slurm_system, tr.test.test_definition.cmd_args_dict
        )
        tr.test.test_template.command_gen_strategy.job_name = Mock(return_value="job_name")

        return (tr, "ucc.sbatch", None)
    elif request.param == "nccl":
        tr = partial_tr(
            name="nccl",
            test=Test(
                test_definition=NCCLTestDefinition(
                    name="nccl", description="nccl", test_template_name="nccl", cmd_args=NCCLCmdArgs()
                ),
                test_template=NcclTest(slurm_system, name="nccl"),
            ),
        )
        tr.test.test_template.command_gen_strategy = NcclTestSlurmCommandGenStrategy(
            slurm_system, tr.test.test_definition.cmd_args_dict
        )
        tr.test.test_template.command_gen_strategy.job_name = Mock(return_value="job_name")

        return (tr, "nccl.sbatch", None)
    elif request.param == "sleep":
        tr = partial_tr(
            name="sleep",
            test=Test(
                test_definition=SleepTestDefinition(
                    name="sleep", description="sleep", test_template_name="sleep", cmd_args=SleepCmdArgs()
                ),
                test_template=Sleep(slurm_system, name="sleep"),
            ),
        )
        tr.test.test_template.command_gen_strategy = SleepSlurmCommandGenStrategy(
            slurm_system, tr.test.test_definition.cmd_args_dict
        )
        tr.test.test_template.command_gen_strategy.job_name = Mock(return_value="job_name")

        return (tr, "sleep.sbatch", None)
    elif request.param.startswith("gpt-"):
        tr = partial_tr(
            name="gpt",
            test=Test(
                test_definition=GPTTestDefinition(
                    name="gpt",
                    description="gpt",
                    test_template_name="gpt",
                    cmd_args=GPTCmdArgs(fdl_config="fdl/config", docker_image_url="https://docker/url"),
                    extra_env_vars={"COMBINE_THRESHOLD": "1"},
                ),
                test_template=JaxToolbox(slurm_system, name="gpt"),
            ),
        )
        tr.test.test_template.command_gen_strategy = JaxToolboxSlurmCommandGenStrategy(
            slurm_system, tr.test.test_definition.cmd_args_dict
        )
        tr.test.test_template.command_gen_strategy.job_name = Mock(return_value="job_name")
        if "pre-test" in request.param:
            pre_test_tr = partial_tr(
                name="nccl",
                test=Test(
                    test_definition=NCCLTestDefinition(
                        name="nccl", description="nccl", test_template_name="nccl", cmd_args=NCCLCmdArgs()
                    ),
                    test_template=NcclTest(slurm_system, name="nccl"),
                ),
            )
            pre_test_tr.test.test_template.command_gen_strategy = NcclTestSlurmCommandGenStrategy(
                slurm_system, pre_test_tr.test.test_definition.cmd_args_dict
            )
            pre_test_tr.test.test_template.command_gen_strategy.job_name = Mock(return_value="job_name")
            tr.pre_test = TestScenario(name=f"{pre_test_tr.name} NCCL pre-test", test_runs=[pre_test_tr])

        return (tr, f"{request.param}.sbatch", "gpt.run")
    elif request.param.startswith("grok-"):
        tr = partial_tr(
            name="grok",
            test=Test(
                test_definition=GrokTestDefinition(
                    name="grok",
                    description="grok",
                    test_template_name="grok",
                    cmd_args=GrokCmdArgs(fdl_config="fdl/config", docker_image_url="https://docker/url"),
                    extra_env_vars={"COMBINE_THRESHOLD": "1"},
                ),
                test_template=JaxToolbox(slurm_system, name="grok"),
            ),
        )
        tr.test.test_template.command_gen_strategy = JaxToolboxSlurmCommandGenStrategy(
            slurm_system, tr.test.test_definition.cmd_args_dict
        )
        tr.test.test_template.command_gen_strategy.job_name = Mock(return_value="job_name")
        if "pre-test" in request.param:
            pre_test_tr = partial_tr(
                name="nccl",
                test=Test(
                    test_definition=NCCLTestDefinition(
                        name="nccl", description="nccl", test_template_name="nccl", cmd_args=NCCLCmdArgs()
                    ),
                    test_template=NcclTest(slurm_system, name="nccl"),
                ),
            )
            pre_test_tr.test.test_template.command_gen_strategy = NcclTestSlurmCommandGenStrategy(
                slurm_system, pre_test_tr.test.test_definition.cmd_args_dict
            )
            pre_test_tr.test.test_template.command_gen_strategy.job_name = Mock(return_value="job_name")
            tr.pre_test = TestScenario(name=f"{pre_test_tr.name} NCCL pre-test", test_runs=[pre_test_tr])

        return (tr, f"{request.param}.sbatch", "grok.run")
    elif request.param == "nemo-launcher":
        tr = partial_tr(
            name="nemo-launcher",
            test=Test(
                test_definition=NeMoLauncherTestDefinition(
                    name="nemo-launcher",
                    description="nemo-launcher",
                    test_template_name="nemo-launcher",
                    cmd_args=NeMoLauncherCmdArgs(),
                ),
                test_template=NeMoLauncher(slurm_system, name="nemo-launcher"),
            ),
        )
        tr.test.test_template.command_gen_strategy = NeMoLauncherSlurmCommandGenStrategy(
            slurm_system, tr.test.test_definition.cmd_args_dict
        )
        tr.test.test_template.command_gen_strategy.job_name = Mock(return_value="job_name")

        return (tr, "nemo-launcher.sbatch", None)

    raise ValueError(f"Unknown test: {request.param}")


def test_sbatch_generation(slurm_system: SlurmSystem, test_req: tuple[TestRun, str]):
    slurm_system.output_path.mkdir(parents=True, exist_ok=True)

    tr = test_req[0]

    sbatch_script = tr.test.test_template.gen_exec_command(tr).split()[-1]
    ref = (Path(__file__).parent / "ref_data" / test_req[1]).read_text().strip()
    if "nemo-launcher" in test_req[1]:
        sbatch_script = slurm_system.output_path / "generated_command.sh"
        ref = ref.replace("__OUTPUT_DIR__", str(slurm_system.output_path.parent))
    else:
        ref = ref.replace("__OUTPUT_DIR__", str(slurm_system.output_path)).replace("__JOB_NAME__", "job_name")

    curr = Path(sbatch_script).read_text().strip()

    assert curr == ref

    run_script = test_req[-1]
    if run_script:
        curr_run_script = Path(slurm_system.output_path / "run.sh").read_text()
        ref_run_script = (Path(__file__).parent / "ref_data" / run_script).read_text()
        assert curr_run_script == ref_run_script
