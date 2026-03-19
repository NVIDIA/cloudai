# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import re
from importlib.metadata import version
from pathlib import Path

import pytest

from cloudai.core import GitRepo, TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.nixl_ep import NixlEPCmdArgs, NixlEPSlurmCommandGenStrategy, NixlEPTestDefinition

EXPANSION_CONTRACTION_PLAN = [
    [0, 1, 2, 3],
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 2, 3, 4, -6, 7],
    [0, 1, 2, 3, 4, 5, 6, 7],
]
EXPANSION_CONTRACTION_PLAN_STR = json.dumps(EXPANSION_CONTRACTION_PLAN)
DOUBLE_EXPANSION_PLAN = [
    [0, 1, 2, 3],
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4, 5, 6, 7],
]
DOUBLE_EXPANSION_PLAN_STR = json.dumps(DOUBLE_EXPANSION_PLAN)


@pytest.fixture
def nixl_ep() -> NixlEPTestDefinition:
    return NixlEPTestDefinition(
        name="nixl_ep",
        description="NIXL Elastic EP benchmark",
        test_template_name="NixlEP",
        cmd_args=NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            elastic_script="/workspace/nixl/examples/device/ep/tests/elastic/elastic.py",
            plan=EXPANSION_CONTRACTION_PLAN_STR,
            num_processes_per_node=[4, 4, 2],
            num_tokens=256,
            num_experts_per_rank=4,
            hidden_dim=8192,
            num_topk=6,
            disable_ll_nvlink=True,
            kineto=True,
            service_startup_timeout_seconds=90,
        ),
        extra_env_vars={
            "LD_LIBRARY_PATH": "/workspace/rdma_core/lib:$LD_LIBRARY_PATH",
        },
    )


@pytest.fixture
def nixl_ep_tr(nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem) -> TestRun:
    return TestRun(name="nixl-ep", num_nodes=3, nodes=[], test=nixl_ep, output_path=slurm_system.output_path)


def normalize_sbatch(content: str, test_run: TestRun, slurm_system: SlurmSystem) -> str:
    normalized = content.replace(str(test_run.output_path.absolute()), "__OUTPUT_DIR__").replace(
        str(slurm_system.install_path.absolute()), "__INSTALL_DIR__"
    )
    normalized = re.sub(r"^#SBATCH --job-name=.*$", "#SBATCH --job-name=__JOB_NAME__", normalized, flags=re.MULTILINE)
    return normalized.replace(version("cloudai"), "__CLOUDAI_VERSION__")


def test_processes_per_node_expands_scalar(nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem) -> None:
    nixl_ep.cmd_args.num_processes_per_node = 5
    test_run = TestRun(name="nixl-ep", num_nodes=2, nodes=[], test=nixl_ep, output_path=slurm_system.output_path)

    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    assert strategy.processes_per_node == [5, 5]


def test_input_json_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not accept `input_json`"):
        NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            input_json="examples/device/ep/tests/elastic/expansion_contraction.json",
            num_processes_per_node=4,
            plan=EXPANSION_CONTRACTION_PLAN_STR,
        )


def test_plan_accepts_single_string_list() -> None:
    cmd_args = NixlEPCmdArgs(
        docker_image_url="docker.io/nvidia/nixl-ep:latest",
        plan=[EXPANSION_CONTRACTION_PLAN_STR],
        num_processes_per_node=4,
    )

    assert cmd_args.plan == EXPANSION_CONTRACTION_PLAN_STR
    assert cmd_args.parse_plan() == EXPANSION_CONTRACTION_PLAN


def test_config_repo_must_not_shadow_container_runtime() -> None:
    with pytest.raises(ValueError, match="must not mount to '/workspace/nixl'"):
        NixlEPTestDefinition(
            name="nixl_ep",
            description="NIXL Elastic EP benchmark",
            test_template_name="NixlEP",
            cmd_args=NixlEPCmdArgs(
                docker_image_url="docker.io/nvidia/nixl-ep:latest",
                plan=EXPANSION_CONTRACTION_PLAN_STR,
                num_processes_per_node=4,
            ),
            git_repos=[
                GitRepo(
                    url="https://github.com/NVIDIA/nixl-configs.git",
                    commit="main",
                    mount_as="/workspace/nixl",
                )
            ],
        )


def test_processes_per_node_rejects_wrong_length(nixl_ep_tr: TestRun, slurm_system: SlurmSystem) -> None:
    nixl_ep_tr.test.cmd_args.num_processes_per_node = [4, 4]
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, nixl_ep_tr)

    with pytest.raises(ValueError, match="must match allocated node count"):
        _ = strategy.processes_per_node


def test_build_elastic_command(nixl_ep_tr: TestRun, slurm_system: SlurmSystem) -> None:
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, nixl_ep_tr)

    master_cmd = strategy.build_elastic_command(4)
    follower_cmd = strategy.build_elastic_command(2, include_tcp_server=True)
    generated_plan_path = nixl_ep_tr.output_path / strategy.GENERATED_PLAN_FILE_NAME

    assert master_cmd == [
        "python3",
        "/workspace/nixl/examples/device/ep/tests/elastic/elastic.py",
        "--plan",
        str(generated_plan_path.absolute()),
        "--num-processes",
        "4",
        "--num-tokens",
        "256",
        "--num-experts-per-rank",
        "4",
        "--hidden-dim",
        "8192",
        "--num-topk",
        "6",
        "--disable-ll-nvlink",
        "--kineto",
    ]
    assert "--tcp-server" not in master_cmd
    assert follower_cmd[-4:] == ["--tcp-server", "$master_ip", "--disable-ll-nvlink", "--kineto"]
    assert json.loads(generated_plan_path.read_text(encoding="utf-8")) == EXPANSION_CONTRACTION_PLAN


def test_build_elastic_command_always_uses_generated_plan_json(slurm_system: SlurmSystem) -> None:
    tdef = NixlEPTestDefinition(
        name="nixl_ep",
        description="NIXL Elastic EP benchmark",
        test_template_name="NixlEP",
        cmd_args=NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            plan=EXPANSION_CONTRACTION_PLAN_STR,
            num_processes_per_node=4,
        ),
    )
    test_run = TestRun(name="nixl-ep", num_nodes=1, nodes=[], test=tdef, output_path=slurm_system.output_path)
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    command = strategy.build_elastic_command(4)

    assert command[1] == "/workspace/nixl/examples/device/ep/tests/elastic/elastic.py"
    assert command[3] == str((test_run.output_path / strategy.GENERATED_PLAN_FILE_NAME).absolute())
    assert json.loads((test_run.output_path / strategy.GENERATED_PLAN_FILE_NAME).read_text(encoding="utf-8")) == (
        EXPANSION_CONTRACTION_PLAN
    )


def test_debug_logging_sets_verbose_env_vars(nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem) -> None:
    nixl_ep.cmd_args.debug_logging = True
    test_run = TestRun(name="nixl-ep", num_nodes=1, nodes=[], test=nixl_ep, output_path=slurm_system.output_path)
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    assert strategy.final_env_vars["PYTHONUNBUFFERED"] == "1"
    assert strategy.final_env_vars["NIXL_DEBUG_LOGGING"] == "yes"
    assert strategy.final_env_vars["NIXL_LOG_LEVEL"] == "TRACE"
    assert strategy.final_env_vars["UCX_LOG_LEVEL"] == "DEBUG"
    assert strategy.final_env_vars["TORCH_DISTRIBUTED_DEBUG"] == "DETAIL"


def test_debug_logging_emits_diagnostics_once_per_node(
    nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem
) -> None:
    nixl_ep.cmd_args.debug_logging = True
    nixl_ep.cmd_args.num_processes_per_node = 10
    test_run = TestRun(name="nixl-ep", num_nodes=1, nodes=[], test=nixl_ep, output_path=slurm_system.output_path)
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    srun_command = strategy.gen_srun_command()

    assert "=== NIXL EP debug diagnostics start ===" in srun_command
    assert "ucx_info -d" in srun_command
    assert "ibv_devinfo -l" in srun_command
    assert "rdma link show" in srun_command
    assert "cat " in srun_command and strategy.GENERATED_PLAN_FILE_NAME in srun_command
    assert ".debug.once" in srun_command


def test_gen_srun_command_single_node(nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem) -> None:
    nixl_ep.cmd_args.num_processes_per_node = 10
    test_run = TestRun(name="nixl-ep", num_nodes=1, nodes=[], test=nixl_ep, output_path=slurm_system.output_path)
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    srun_command = strategy.gen_srun_command()

    assert "wait_for_master_services" not in srun_command
    assert "wait_for_phase_completion()" in srun_command
    assert "Waiting for phase 0 before starting wave 1" in srun_command
    assert "Waiting for phase 2 before starting wave 2" in srun_command
    assert srun_command.count("--num-processes 4") == 2
    assert srun_command.count("--num-processes 2") == 1
    assert srun_command.count("--tcp-server $master_ip") == 2
    assert srun_command.count("--open-mode=append") == 2
    assert "--nodelist=$SLURM_JOB_MASTER_NODE" in srun_command
    assert "--relative=1" not in srun_command


def test_single_node_launch_waves_follow_plan(nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem) -> None:
    nixl_ep.cmd_args.num_processes_per_node = 10
    test_run = TestRun(name="nixl-ep", num_nodes=1, nodes=[], test=nixl_ep, output_path=slurm_system.output_path)
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    assert strategy.single_node_launch_waves == [(None, 4), (0, 4), (2, 2)]


def test_single_node_launch_waves_follow_double_expansion_public_plan(
    nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem
) -> None:
    nixl_ep.cmd_args.plan = DOUBLE_EXPANSION_PLAN_STR
    nixl_ep.cmd_args.num_processes_per_node = 8
    test_run = TestRun(name="nixl-ep", num_nodes=1, nodes=[], test=nixl_ep, output_path=slurm_system.output_path)
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    assert strategy.single_node_launch_waves == [(None, 4), (0, 2), (1, 2)]


def test_single_node_launch_waves_reject_scalar_mismatch(
    nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem
) -> None:
    nixl_ep.cmd_args.num_processes_per_node = 9
    test_run = TestRun(name="nixl-ep", num_nodes=1, nodes=[], test=nixl_ep, output_path=slurm_system.output_path)
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    with pytest.raises(ValueError, match="launch waves total \\(10\\), got 9"):
        _ = strategy.single_node_launch_waves


def test_gen_exec_command_matches_reference(nixl_ep_tr: TestRun, slurm_system: SlurmSystem) -> None:
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, nixl_ep_tr)

    sbatch_cmd = strategy.gen_exec_command()

    assert sbatch_cmd == f"sbatch {nixl_ep_tr.output_path / 'cloudai_sbatch_script.sh'}"

    content = (nixl_ep_tr.output_path / "cloudai_sbatch_script.sh").read_text().strip()
    content = normalize_sbatch(content, nixl_ep_tr, slurm_system)

    ref = (Path(__file__).parents[2] / "ref_data" / "nixl-ep.sbatch").read_text().strip()
    assert content == ref
