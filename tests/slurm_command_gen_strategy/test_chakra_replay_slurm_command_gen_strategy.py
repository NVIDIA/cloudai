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

import re
from pathlib import Path
from typing import Any, Dict, List, Union
from unittest.mock import Mock

import pytest

from cloudai import GitRepo, PythonExecutable, TestRun
from cloudai._core.test import Test
from cloudai.systems import SlurmSystem
from cloudai.workloads.chakra_replay import (
    ChakraReplayCmdArgs,
    ChakraReplaySlurmCommandGenStrategy,
    ChakraReplayTestDefinition,
)


@pytest.fixture
def cmd_gen_strategy(slurm_system: SlurmSystem) -> ChakraReplaySlurmCommandGenStrategy:
    return ChakraReplaySlurmCommandGenStrategy(slurm_system, {})


@pytest.fixture
def chakra_replay_tr(cmd_args: Dict[str, Any]) -> TestRun:
    chakra = ChakraReplayTestDefinition(
        name="name",
        description="desc",
        test_template_name="template",
        cmd_args=ChakraReplayCmdArgs(
            docker_image_url=cmd_args.get("docker_image_url", ""),
            trace_dir=cmd_args.get("trace_dir", ""),
            warmup_iters=cmd_args.get("warmup_iters", 0),
            iters=cmd_args.get("iters", 10),
            reuse_tensors=cmd_args.get("reuse_tensors", True),
            backend_name=cmd_args.get("backend.name", "pytorch-dist"),
            profiler_enabled=cmd_args.get("profiler.enabled", False),
            log_level=cmd_args.get("logging.level", "INFO"),
        ),
        comm_replay_executable=PythonExecutable(git_repo=GitRepo(url="./git_repo", commit="commit")),
    )
    test = Test(test_definition=chakra, test_template=Mock())

    chakra.comm_replay_executable.git_repo.installed_path = Path("/git_repo_path")
    return TestRun(name="chakra_replay_tr", test=test, nodes=[], num_nodes=1)


@pytest.mark.parametrize(
    "job_name_prefix, env_vars, cmd_args, expected_result",
    [
        (
            "chakra_replay",
            {"NCCL_DEBUG": "INFO"},
            {"docker_image_url": "fake_image_url", "trace_dir": "/output/traces/"},
            {
                "image_path": "fake_image_url",
                "container_mounts": ["/output/traces/:/output/traces/,/git_repo_path:/git_repo_path"],
            },
        ),
        (
            "chakra_replay",
            {"NCCL_DEBUG": "INFO"},
            {
                "docker_image_url": "another_image_url",
                "trace_dir": "/another/trace_dir/",
            },
            {
                "image_path": "another_image_url",
                "container_mounts": ["/another/trace_dir/:/another/trace_dir/,/git_repo_path:/git_repo_path"],
            },
        ),
    ],
)
def test_parse_slurm_args(
    cmd_gen_strategy: ChakraReplaySlurmCommandGenStrategy,
    job_name_prefix: str,
    env_vars: Dict[str, Union[str, List[str]]],
    cmd_args: Dict[str, Any],
    expected_result: Dict[str, Any],
    chakra_replay_tr: TestRun,
) -> None:
    slurm_args = cmd_gen_strategy._parse_slurm_args(job_name_prefix, env_vars, cmd_args, chakra_replay_tr)
    assert slurm_args["image_path"] == expected_result["image_path"]
    assert cmd_gen_strategy._container_mounts(chakra_replay_tr) == expected_result["container_mounts"]


@pytest.mark.parametrize(
    "cmd_args, num_nodes, ntasks_per_node",
    [
        (
            {
                "trace_dir": "/output/traces/",
                "warmup_iters": 5,
                "iters": 10,
                "reuse_tensors": False,
                "backend.name": "pytorch-dist",
                "backend.backend": "nccl",
                "profiler.enabled": False,
                "logging.level": "INFO",
            },
            2,
            4,
        ),
    ],
)
def test_generate_srun_command(
    cmd_gen_strategy: ChakraReplaySlurmCommandGenStrategy,
    cmd_args: Dict[str, Any],
    num_nodes: int,
    ntasks_per_node: int,
    chakra_replay_tr: TestRun,
    tmp_path: Path,
) -> None:
    chakra_replay_tr.num_nodes = num_nodes
    cmd_gen_strategy.system.ntasks_per_node = ntasks_per_node
    chakra_replay_tr.output_path = tmp_path

    slurm_args = cmd_gen_strategy._parse_slurm_args("test", {}, {}, chakra_replay_tr)
    command = cmd_gen_strategy._gen_srun_command(slurm_args, {}, {}, chakra_replay_tr)

    generated_commands = command.strip().split("\n")
    assert len(generated_commands) == 2

    timestamp_pattern = r"\d{14}"
    container_name_pattern = re.compile(r"--container-name=chakra_replay_container_" + timestamp_pattern)
    assert container_name_pattern.search(generated_commands[0]) is not None
    assert container_name_pattern.search(generated_commands[1]) is not None

    assert f"-N {num_nodes}" in generated_commands[0]
    assert f"-n {num_nodes}" in generated_commands[0]
    assert "--ntasks-per-node=1" in generated_commands[0]

    assert f"-N {num_nodes}" in generated_commands[1]
    assert f"-n {num_nodes * ntasks_per_node}" in generated_commands[1]
    assert f"--ntasks-per-node={ntasks_per_node}" in generated_commands[1]
    assert "comm_replay --config" in generated_commands[1]
