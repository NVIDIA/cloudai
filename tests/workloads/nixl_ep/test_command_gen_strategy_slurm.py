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

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.nixl_ep import (
    NixlEPCmdArgs,
    NixlEPSlurmCommandGenStrategy,
    NixlEPTestDefinition,
)
from cloudai.workloads.nixl_ep.nixl_ep import GENERATED_PLAN_FILE_NAME
from cloudai.workloads.nixl_ep.slurm_command_gen_strategy import NixlEPLaunch

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
SINGLE_EXPANSION_PLAN = [
    [0, 1, 2, 3],
    [0, 1, 2, 3, 4, 5, 6, 7],
]
SINGLE_EXPANSION_PLAN_STR = json.dumps(SINGLE_EXPANSION_PLAN)
SINGLE_RANK_PLAN_STR = json.dumps([[0]])


def make_cmd_args(**overrides: object) -> NixlEPCmdArgs:
    payload = {
        "docker_image_url": "docker.io/nvidia/nixl-ep:latest",
        "elastic_script": "/workspace/nixl/examples/device/ep/tests/elastic/elastic.py",
        "plan": EXPANSION_CONTRACTION_PLAN_STR,
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
    payload.update(overrides)
    return NixlEPCmdArgs.model_validate(payload)


def replace_cmd_args(cmd_args: NixlEPCmdArgs, **overrides: object) -> NixlEPCmdArgs:
    payload = cmd_args.model_dump()
    payload.update(cmd_args.model_extra or {})
    payload.update(overrides)
    return NixlEPCmdArgs.model_validate(payload)


@pytest.fixture
def nixl_ep() -> NixlEPTestDefinition:
    return NixlEPTestDefinition(
        name="nixl_ep",
        description="NIXL Elastic EP benchmark",
        test_template_name="NixlEP",
        cmd_args=make_cmd_args(),
        extra_env_vars={
            "LD_LIBRARY_PATH": "/workspace/rdma_core/lib:$LD_LIBRARY_PATH",
        },
    )


@pytest.fixture
def nixl_ep_tr(nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem) -> TestRun:
    return TestRun(
        name="nixl-ep",
        num_nodes=3,
        nodes=[],
        test=nixl_ep,
        output_path=slurm_system.output_path,
    )


def normalize_sbatch(content: str, test_run: TestRun, slurm_system: SlurmSystem) -> str:
    normalized = content.replace(str(slurm_system.install_path.absolute()), "__INSTALL_DIR__").replace(
        str(test_run.output_path.parent.absolute()), "__OUTPUT_DIR__"
    )
    normalized = re.sub(
        r"^#SBATCH --job-name=.*$",
        "#SBATCH --job-name=__JOB_NAME__",
        normalized,
        flags=re.MULTILINE,
    )
    return normalized.replace(version("cloudai"), "__CLOUDAI_VERSION__")


def significant_sbatch_lines(content: str) -> list[str]:
    return [line for line in content.splitlines() if line.strip() and not line.lstrip().startswith("echo ")]


def normalize_stages(strategy: NixlEPSlurmCommandGenStrategy) -> list[tuple[int, tuple[int, ...]]]:
    num_nodes, _ = strategy.get_cached_nodes_spec()
    normalized_stages: list[tuple[int, tuple[int, ...]]] = []
    for stage in strategy.plan_stages:
        per_node_processes = [0] * num_nodes
        for launch in stage.launches:
            per_node_processes[launch.node_idx] = launch.num_processes
        normalized_stages.append((stage.idx, tuple(per_node_processes)))
    return normalized_stages


def test_num_processes_per_node_returns_integer(
    nixl_ep: NixlEPTestDefinition,
    slurm_system: SlurmSystem,
) -> None:
    nixl_ep.cmd_args.num_processes_per_node = 5
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=2,
        nodes=[],
        test=nixl_ep,
        output_path=slurm_system.output_path,
    )

    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    assert strategy.num_processes_per_node == 5


def test_missing_plan_is_rejected() -> None:
    with pytest.raises(ValueError, match="Field required"):
        NixlEPCmdArgs.model_validate(
            {
                "docker_image_url": "docker.io/nvidia/nixl-ep:latest",
                "num_processes_per_node": 4,
            }
        )


def test_plan_accepts_single_string_list() -> None:
    cmd_args = NixlEPCmdArgs(
        docker_image_url="docker.io/nvidia/nixl-ep:latest",
        plan=[EXPANSION_CONTRACTION_PLAN_STR],
        num_processes_per_node=4,
    )

    assert cmd_args.plan == [EXPANSION_CONTRACTION_PLAN_STR]


def test_plan_accepts_multiple_strings_for_dse() -> None:
    cmd_args = NixlEPCmdArgs(
        docker_image_url="docker.io/nvidia/nixl-ep:latest",
        plan=[EXPANSION_CONTRACTION_PLAN_STR, DOUBLE_EXPANSION_PLAN_STR],
        num_processes_per_node=4,
    )

    assert cmd_args.plan == [EXPANSION_CONTRACTION_PLAN_STR, DOUBLE_EXPANSION_PLAN_STR]


def test_plan_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="plan list must not be empty"):
        NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            plan=[],
            num_processes_per_node=4,
        )


def test_plan_rejects_list_with_empty_string() -> None:
    with pytest.raises(ValueError, match="plan list must not contain empty strings"):
        NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            plan=[EXPANSION_CONTRACTION_PLAN_STR, "  "],
            num_processes_per_node=4,
        )


def test_plan_rejects_list_with_invalid_json() -> None:
    with pytest.raises(ValueError, match="plan must be valid JSON"):
        NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            plan=[EXPANSION_CONTRACTION_PLAN_STR, "not-json"],
            num_processes_per_node=4,
        )


def test_plan_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="plan must be valid JSON"):
        NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            plan="not-json",
            num_processes_per_node=4,
        )


def test_plan_rejects_non_integer_ranks() -> None:
    with pytest.raises(ValueError, match="Each plan rank must be an integer"):
        NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            plan='[[0, "1"]]',
            num_processes_per_node=2,
        )


def test_num_processes_per_node_rejects_list(nixl_ep_tr: TestRun, slurm_system: SlurmSystem) -> None:
    nixl_ep_tr.test.cmd_args.num_processes_per_node = [4, 4, 2]
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, nixl_ep_tr)

    with pytest.raises(ValueError, match="requires num_processes_per_node to be an integer"):
        _ = strategy.num_processes_per_node


def test_build_benchmark_command(nixl_ep_tr: TestRun, slurm_system: SlurmSystem) -> None:
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, nixl_ep_tr)

    master_cmd = strategy._build_benchmark_command(NixlEPLaunch(node_idx=0, num_processes=4, include_tcp_server=False))
    follower_cmd = strategy._build_benchmark_command(NixlEPLaunch(node_idx=0, num_processes=2, include_tcp_server=True))
    generated_plan_path = nixl_ep_tr.output_path / GENERATED_PLAN_FILE_NAME

    assert master_cmd == [
        "python3",
        "/workspace/nixl/examples/device/ep/tests/elastic/elastic.py",
        "--plan",
        str(generated_plan_path.absolute()),
        "--num-processes",
        "4",
        "--disable-ll-nvlink",
        "--hidden-dim",
        "8192",
        "--kineto",
        "--num-experts-per-rank",
        "4",
        "--num-tokens",
        "256",
        "--num-topk",
        "6",
    ]
    assert "--tcp-server" not in master_cmd
    assert follower_cmd[6:10] == [
        "--tcp-server",
        "$master_ip",
        "--disable-ll-nvlink",
        "--hidden-dim",
    ]
    assert "--service-startup-timeout-seconds" not in follower_cmd
    strategy._write_plan_file()
    assert json.loads(generated_plan_path.read_text(encoding="utf-8")) == EXPANSION_CONTRACTION_PLAN


def test_build_benchmark_command_always_uses_generated_plan_json(
    slurm_system: SlurmSystem,
) -> None:
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
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=tdef,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    command = strategy._build_benchmark_command(NixlEPLaunch(node_idx=0, num_processes=4, include_tcp_server=False))

    assert command[1] == "/workspace/nixl/examples/device/ep/tests/elastic/elastic.py"
    assert command[3] == str((test_run.output_path / GENERATED_PLAN_FILE_NAME).absolute())
    strategy._write_plan_file()
    assert json.loads((test_run.output_path / GENERATED_PLAN_FILE_NAME).read_text(encoding="utf-8")) == (
        EXPANSION_CONTRACTION_PLAN
    )


def test_relative_elastic_script_is_resolved_under_container_runtime_root() -> None:
    tdef = NixlEPTestDefinition(
        name="nixl_ep",
        description="NIXL Elastic EP benchmark",
        test_template_name="NixlEP",
        cmd_args=NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            elastic_script="examples/device/ep/tests/elastic/elastic.py",
            plan=EXPANSION_CONTRACTION_PLAN_STR,
            num_processes_per_node=4,
        ),
    )

    assert tdef.installables == [tdef.docker_image]


def test_build_benchmark_command_passes_through_relative_elastic_script_path(
    slurm_system: SlurmSystem,
) -> None:
    tdef = NixlEPTestDefinition(
        name="nixl_ep",
        description="NIXL Elastic EP benchmark",
        test_template_name="NixlEP",
        cmd_args=NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            elastic_script="examples/device/ep/tests/elastic/elastic.py",
            plan=EXPANSION_CONTRACTION_PLAN_STR,
            num_processes_per_node=4,
        ),
    )
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=tdef,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    command = strategy._build_benchmark_command(NixlEPLaunch(node_idx=0, num_processes=4, include_tcp_server=False))

    assert command[1] == "examples/device/ep/tests/elastic/elastic.py"


def test_build_benchmark_command_omits_disable_ll_nvlink_by_default(
    slurm_system: SlurmSystem,
) -> None:
    tdef = NixlEPTestDefinition(
        name="nixl_ep",
        description="NIXL Elastic EP benchmark",
        test_template_name="NixlEP",
        cmd_args=NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            plan=DOUBLE_EXPANSION_PLAN_STR,
            num_processes_per_node=8,
        ),
    )
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=tdef,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    command = strategy._build_benchmark_command(NixlEPLaunch(node_idx=0, num_processes=4, include_tcp_server=False))

    assert "--disable-ll-nvlink" not in command


def test_build_benchmark_command_passes_through_extra_flags(
    slurm_system: SlurmSystem,
) -> None:
    tdef = NixlEPTestDefinition(
        name="nixl_ep",
        description="NIXL Elastic EP benchmark",
        test_template_name="NixlEP",
        cmd_args=NixlEPCmdArgs.model_validate(
            {
                "docker_image_url": "docker.io/nvidia/nixl-ep:latest",
                "plan": DOUBLE_EXPANSION_PLAN_STR,
                "num_processes_per_node": 8,
                "service_startup_timeout_seconds": 90,
                "store_port": 9999,
                "dry_run": True,
                "custom_arg": "value",
                "ignored_arg": None,
            }
        ),
    )
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=tdef,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    command = strategy._build_benchmark_command(NixlEPLaunch(node_idx=0, num_processes=4, include_tcp_server=True))

    assert "--dry-run" in command
    assert "--custom-arg" in command
    assert "value" in command
    assert "--service-startup-timeout-seconds" not in command
    assert "--store-port" not in command
    assert "--ignored-arg" not in command


def test_wait_for_master_services_only_probes_tcpstore(
    nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem
) -> None:
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=2,
        nodes=[],
        test=nixl_ep,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    wait_function = strategy.generate_wait_for_master_services_function()

    assert f"/dev/tcp/$master_ip/{nixl_ep.cmd_args.store_port}" in wait_function
    assert "/dev/tcp/$master_ip/10000" not in wait_function


def test_phase_transition_timeout_divides_job_timeout_by_plan_phases(
    nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem
) -> None:
    nixl_ep.cmd_args.plan = SINGLE_EXPANSION_PLAN_STR
    nixl_ep.cmd_args.num_processes_per_node = 8
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=nixl_ep,
        output_path=slurm_system.output_path,
        time_limit="00:10:00",
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    assert strategy.phase_transition_timeout_seconds == 300
    assert "local timeout=300" in strategy.generate_wait_for_phase_completion_function()


def test_phase_transition_timeout_divides_default_budget_without_job_timeout(
    nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem
) -> None:
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=nixl_ep,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    assert strategy.phase_transition_timeout_seconds == 150
    assert "local timeout=150" in strategy.generate_wait_for_phase_completion_function()


def test_gen_srun_command_single_node(nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem) -> None:
    nixl_ep.cmd_args.num_processes_per_node = 10
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=nixl_ep,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    srun_command = strategy.gen_srun_command()

    assert "wait_for_master_services" not in srun_command
    assert "wait_for_phase_completion()" in srun_command
    assert 'wait_for_phase_completion "0"' in srun_command
    assert 'wait_for_phase_completion "2"' in srun_command
    assert srun_command.count("--num-processes 4") == 2
    assert srun_command.count("--num-processes 2") == 1
    assert srun_command.count("--tcp-server $master_ip") == 2
    assert srun_command.count("--open-mode=append") == 2
    assert "--nodelist=$SLURM_JOB_MASTER_NODE" in srun_command
    assert "--relative=1" not in srun_command


def test_gen_srun_command_single_node_static_plan(nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem) -> None:
    nixl_ep.cmd_args.plan = json.dumps([[0, 1, 2, 3]])
    nixl_ep.cmd_args.num_processes_per_node = 4
    nixl_ep.cmd_args = replace_cmd_args(nixl_ep.cmd_args, disable_ll_nvlink=False)
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=nixl_ep,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    srun_command = strategy.gen_srun_command()

    assert "wait_for_phase_completion()" not in srun_command
    assert srun_command.count("--num-processes 4") == 1
    assert "--disable-ll-nvlink" not in srun_command


def test_gen_srun_command_single_node_single_rank_plan(
    nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem
) -> None:
    nixl_ep.cmd_args.plan = SINGLE_RANK_PLAN_STR
    nixl_ep.cmd_args.num_processes_per_node = 1
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=nixl_ep,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    srun_command = strategy.gen_srun_command()

    assert "wait_for_master_services()" not in srun_command
    assert "wait_for_phase_completion()" not in srun_command
    assert srun_command.count("--num-processes 1") == 1
    assert "--tcp-server $master_ip" not in srun_command
    assert "--open-mode=append" not in srun_command


def test_gen_srun_command_rejects_process_list(slurm_system: SlurmSystem) -> None:
    tdef = NixlEPTestDefinition(
        name="nixl_ep",
        description="NIXL Elastic EP benchmark",
        test_template_name="NixlEP",
        cmd_args=NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            plan=json.dumps([[0, 1, 2, 3]]),
            num_processes_per_node=[4],
        ),
    )
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=tdef,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    with pytest.raises(ValueError, match="requires num_processes_per_node to be an integer"):
        strategy.gen_srun_command()


def test_single_node_stages_cover_each_plan_phase(nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem) -> None:
    nixl_ep.cmd_args.num_processes_per_node = 10
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=nixl_ep,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    assert normalize_stages(strategy) == [(0, (4,)), (1, (4,)), (2, (0,)), (3, (2,))]


def test_single_node_stages_follow_double_expansion_public_plan(
    nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem
) -> None:
    nixl_ep.cmd_args.plan = DOUBLE_EXPANSION_PLAN_STR
    nixl_ep.cmd_args.num_processes_per_node = 8
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=nixl_ep,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    assert normalize_stages(strategy) == [(0, (4,)), (1, (2,)), (2, (2,))]


def test_single_node_stages_follow_single_expansion_public_plan(
    nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem
) -> None:
    nixl_ep.cmd_args.plan = SINGLE_EXPANSION_PLAN_STR
    nixl_ep.cmd_args.num_processes_per_node = 8
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=nixl_ep,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    assert normalize_stages(strategy) == [(0, (4,)), (1, (4,))]


def test_single_node_single_stage_plan_has_one_stage(nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem) -> None:
    nixl_ep.cmd_args.plan = json.dumps([[0, 1, 2, 3]])
    nixl_ep.cmd_args.num_processes_per_node = 4
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=nixl_ep,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    assert normalize_stages(strategy) == [(0, (4,))]


def test_multi_node_stages_match_public_two_node_single_expansion(
    slurm_system: SlurmSystem,
) -> None:
    tdef = NixlEPTestDefinition(
        name="nixl_ep",
        description="NIXL Elastic EP benchmark",
        test_template_name="NixlEP",
        cmd_args=NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            plan=SINGLE_EXPANSION_PLAN_STR,
            num_processes_per_node=4,
        ),
    )
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=2,
        nodes=[],
        test=tdef,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    assert normalize_stages(strategy) == [(0, (4, 0)), (1, (0, 4))]


def test_multi_node_single_stage_plan_splits_initial_launches_across_nodes(
    slurm_system: SlurmSystem,
) -> None:
    tdef = NixlEPTestDefinition(
        name="nixl_ep",
        description="NIXL Elastic EP benchmark",
        test_template_name="NixlEP",
        cmd_args=NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            plan=json.dumps([[0, 1, 2, 3, 4, 5, 6, 7]]),
            num_processes_per_node=4,
        ),
    )
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=2,
        nodes=[],
        test=tdef,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    assert normalize_stages(strategy) == [(0, (4, 4))]


def test_gen_srun_command_single_node_double_expansion_omits_disable_flag(
    nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem
) -> None:
    nixl_ep.cmd_args.plan = DOUBLE_EXPANSION_PLAN_STR
    nixl_ep.cmd_args.num_processes_per_node = 8
    nixl_ep.cmd_args = replace_cmd_args(nixl_ep.cmd_args, disable_ll_nvlink=False)
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=nixl_ep,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    srun_command = strategy.gen_srun_command()

    assert 'wait_for_phase_completion "0"' in srun_command
    assert 'wait_for_phase_completion "1"' in srun_command
    assert srun_command.count("--num-processes 2") == 2
    assert "--disable-ll-nvlink" not in srun_command


def test_gen_srun_command_multi_node_public_single_expansion_waits_for_phase_before_second_stage(
    slurm_system: SlurmSystem,
) -> None:
    tdef = NixlEPTestDefinition(
        name="nixl_ep",
        description="NIXL Elastic EP benchmark",
        test_template_name="NixlEP",
        cmd_args=NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            plan=SINGLE_EXPANSION_PLAN_STR,
            num_processes_per_node=4,
        ),
    )
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=2,
        nodes=[],
        test=tdef,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    srun_command = strategy.gen_srun_command()

    assert "wait_for_master_services()" in srun_command
    assert "wait_for_phase_completion()" in srun_command
    assert 'wait_for_phase_completion "0"' in srun_command
    assert srun_command.count("--num-processes 4") == 2
    assert srun_command.count("--relative=1") == 1
    assert srun_command.count("--nodelist=$SLURM_JOB_MASTER_NODE") == 1
    assert srun_command.count("--tcp-server $master_ip") == 1
    assert srun_command.count("--open-mode=append") == 1


def test_gen_srun_command_multi_node_single_stage_starts_followers(
    slurm_system: SlurmSystem,
) -> None:
    tdef = NixlEPTestDefinition(
        name="nixl_ep",
        description="NIXL Elastic EP benchmark",
        test_template_name="NixlEP",
        cmd_args=NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            plan=json.dumps([[0, 1, 2, 3, 4, 5, 6, 7]]),
            num_processes_per_node=4,
        ),
    )
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=2,
        nodes=[],
        test=tdef,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    srun_command = strategy.gen_srun_command()

    assert "wait_for_master_services()" in srun_command
    assert "wait_for_phase_completion()" not in srun_command
    assert srun_command.count("--num-processes 4") == 2
    assert srun_command.count("--relative=1") == 1
    assert srun_command.count("--tcp-server $master_ip") == 1


def test_single_node_stages_reject_mismatch(nixl_ep: NixlEPTestDefinition, slurm_system: SlurmSystem) -> None:
    nixl_ep.cmd_args.num_processes_per_node = 9
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=nixl_ep,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    with pytest.raises(ValueError, match="total launched workers \\(10\\), got 9"):
        _ = strategy.plan_stages


def test_gen_srun_command_single_launch_reports_success(
    slurm_system: SlurmSystem,
) -> None:
    tdef = NixlEPTestDefinition(
        name="nixl_ep",
        description="NIXL Elastic EP benchmark",
        test_template_name="NixlEP",
        cmd_args=NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            plan=SINGLE_RANK_PLAN_STR,
            num_processes_per_node=1,
        ),
    )
    test_run = TestRun(
        name="nixl-ep",
        num_nodes=1,
        nodes=[],
        test=tdef,
        output_path=slurm_system.output_path,
    )
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, test_run)

    srun_command = strategy.gen_srun_command()

    assert 'echo "All NIXL EP launches completed successfully"' in srun_command
    assert 'if [ "$rc" -eq 0 ]; then' in srun_command
    assert "exit $rc" in srun_command


def test_gen_exec_command_matches_reference(nixl_ep_tr: TestRun, slurm_system: SlurmSystem) -> None:
    slurm_system.container_mount_home = True
    strategy = NixlEPSlurmCommandGenStrategy(slurm_system, nixl_ep_tr)

    sbatch_cmd = strategy.gen_exec_command()

    assert sbatch_cmd == f"sbatch {nixl_ep_tr.output_path / 'cloudai_sbatch_script.sh'}"

    content = (nixl_ep_tr.output_path / "cloudai_sbatch_script.sh").read_text().strip()
    content = normalize_sbatch(content, nixl_ep_tr, slurm_system)

    ref = (Path(__file__).parents[2] / "ref_data" / "nixl-ep.sbatch").read_text().strip()
    ref = normalize_sbatch(ref, nixl_ep_tr, slurm_system)
    assert significant_sbatch_lines(content) == significant_sbatch_lines(ref)
