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
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union
from unittest.mock import Mock

import pytest
import toml

from cloudai import GitRepo, TestRun
from cloudai._core.test import Test
from cloudai.systems import SlurmSystem
from cloudai.workloads.chakra_replay import (
    ChakraReplayCmdArgs,
    ChakraReplayConfigParser,
    ChakraReplaySlurmCommandGenStrategy,
    ChakraReplayTestDefinition,
)


@pytest.fixture
def cmd_gen_strategy(slurm_system: SlurmSystem) -> ChakraReplaySlurmCommandGenStrategy:
    return ChakraReplaySlurmCommandGenStrategy(slurm_system, {})


@pytest.fixture
def test_run(cmd_args: Dict[str, Any]) -> TestRun:
    chakra = ChakraReplayTestDefinition(
        name="name",
        description="desc",
        test_template_name="template",
        cmd_args=ChakraReplayCmdArgs(
            docker_image_url=cmd_args.get("docker_image_url", ""),
            trace_dir=cmd_args.get("trace_dir", ""),
            git_repo=GitRepo(url="./git_repo", commit="commit"),
            warmup_iters=cmd_args.get("warmup_iters", 0),
            iters=cmd_args.get("iters", 10),
        ),
    )
    test = Test(test_definition=chakra, test_template=Mock())

    chakra.comm_replay_executable.git_repo.installed_path = Path("/git_repo_path")

    return TestRun(name="test_run", test=test, nodes=[], num_nodes=1)


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
            {"docker_image_url": "another_image_url", "trace_dir": "/another/trace_dir/"},
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
    test_run: TestRun,
) -> None:
    slurm_args = cmd_gen_strategy._parse_slurm_args(job_name_prefix, env_vars, cmd_args, test_run)
    assert slurm_args["image_path"] == expected_result["image_path"]
    assert cmd_gen_strategy._container_mounts(test_run) == expected_result["container_mounts"]


@pytest.mark.parametrize(
    "cmd_args, expected_config",
    [
        (
            {"trace_dir": "/output/traces/", "warmup_iters": 5, "iters": 10},
            {
                "trace": {"directory": "/output/traces/"},
                "run": {"warmup_iters": 5, "iters": 10},
                "comm": {"backend": {"name": "pytorch-dist", "backend": "nccl"}},
                "tensor_allocator": {"reuse_tensors": False},
                "logging": {"level": "INFO"},
                "profiler": {"enabled": False},
            },
        ),
        (
            {
                "trace_dir": "/data/traces/",
                "warmup_iters": 0,
                "iters": 50,
                "profiler.enabled": True,
                "backend.name": "custom_backend",
                "logging.level": "DEBUG",
                "git_repo.url": "https://example.com/repo.git",
                "git_repo.commit": "abc123",
            },
            {
                "trace": {"directory": "/data/traces/"},
                "run": {"warmup_iters": 0, "iters": 50},
                "profiler": {"enabled": True},
                "comm": {"backend": {"name": "custom_backend", "backend": "nccl"}},
                "logging": {"level": "DEBUG"},
                "tensor_allocator": {"reuse_tensors": False},
            },
        ),
    ],
)
def test_write_toml_config(cmd_args: Dict[str, Any], expected_config: Dict[str, Any]) -> None:
    config_parser = ChakraReplayConfigParser(cmd_args)

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = config_parser.write_to_toml(Path(temp_dir))

        with config_path.open("r") as toml_file:
            config_data = toml.load(toml_file)

    assert config_data == expected_config


@pytest.mark.parametrize(
    "cmd_args, num_nodes, ntasks_per_node",
    [
        (
            {"trace_dir": "/output/traces/"},
            2,
            4,
        ),
        (
            {"trace_dir": "/another/path/"},
            3,
            2,
        ),
    ],
)
def test_generate_srun_command(
    cmd_gen_strategy: ChakraReplaySlurmCommandGenStrategy,
    cmd_args: Dict[str, Any],
    num_nodes: int,
    ntasks_per_node: int,
    test_run: TestRun,
) -> None:
    test_run.num_nodes = num_nodes
    cmd_gen_strategy.system.ntasks_per_node = ntasks_per_node

    with tempfile.TemporaryDirectory() as temp_dir:
        test_run.output_path = Path(temp_dir)
        slurm_args = cmd_gen_strategy._parse_slurm_args("test", {}, cmd_args, test_run)
        command = cmd_gen_strategy._gen_srun_command(slurm_args, {}, cmd_args, test_run)

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
