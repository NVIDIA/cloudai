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

from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock

import pytest
import toml

from cloudai import GitRepo, TestRun
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
    return TestRun(name="test_run", test=test, nodes=["node1"], num_nodes=1)


@pytest.mark.parametrize(
    "job_name_prefix, env_vars, cmd_args, expected_result",
    [
        (
            "chakra_replay",
            {"NCCL_DEBUG": "INFO"},
            {"docker_image_url": "fake_image_url", "trace_dir": "/output/traces/"},
            {"image_path": "fake_image_url", "container_mounts": ["/output/traces/:/output/traces/"]},
        ),
        (
            "chakra_replay",
            {"NCCL_DEBUG": "INFO"},
            {"docker_image_url": "another_image_url", "trace_dir": "/another/trace_dir/"},
            {"image_path": "another_image_url", "container_mounts": ["/another/trace_dir/:/another/trace_dir/"]},
        ),
    ],
)
def test_parse_slurm_args(
    cmd_gen_strategy: ChakraReplaySlurmCommandGenStrategy,
    job_name_prefix: str,
    env_vars: Dict[str, str],
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
            },
        ),
        (
            {"trace_dir": "/another/path/", "warmup_iters": 2, "iters": 20},
            {
                "trace": {"directory": "/another/path/"},
                "run": {"warmup_iters": 2, "iters": 20},
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
                "comm": {"backend": {"name": "custom_backend"}},
                "logging": {"level": "DEBUG"},
                "git_repo": {"url": "https://example.com/repo.git", "commit": "abc123"},
            },
        ),
        (
            {
                "trace_dir": "/minimal/path/",
            },
            {
                "trace": {"directory": "/minimal/path/"},
            },
        ),
        (
            {
                "trace_dir": "/nested/config/",
                "profiler.enabled": False,
                "backend.name": "default",
                "logging.level": "INFO",
            },
            {
                "trace": {"directory": "/nested/config/"},
                "profiler": {"enabled": False},
                "comm": {"backend": {"name": "default"}},
                "logging": {"level": "INFO"},
            },
        ),
        (
            {
                "trace_dir": "/complex/params/",
                "warmup_iters": 3,
                "iters": 15,
                "backend.name": "advanced_backend",
                "git_repo.url": "https://git.com/another_repo.git",
                "git_repo.commit": "def456",
            },
            {
                "trace": {"directory": "/complex/params/"},
                "run": {"warmup_iters": 3, "iters": 15},
                "comm": {"backend": {"name": "advanced_backend"}},
                "git_repo": {"url": "https://git.com/another_repo.git", "commit": "def456"},
            },
        ),
    ],
)
def test_write_toml_config(
    cmd_gen_strategy: ChakraReplaySlurmCommandGenStrategy,
    cmd_args: Dict[str, Any],
    expected_config: Dict[str, Any],
    test_run: TestRun,
) -> None:
    config_path = Path(cmd_gen_strategy._write_toml_config(cmd_args, test_run))

    with config_path.open("r") as toml_file:
        config_data = toml.load(toml_file)

    assert config_data == expected_config


@pytest.mark.parametrize(
    "cmd_args, expected_command",
    [
        (
            {"trace_dir": "/output/traces/", "warmup_iters": 5, "iters": 10},
            ["comm_replay", "--config /output/config.toml"],
        ),
        (
            {"trace_dir": "/another/path/"},
            ["comm_replay", "--config /output/config.toml"],
        ),
    ],
)
def test_generate_test_command(
    cmd_gen_strategy: ChakraReplaySlurmCommandGenStrategy,
    cmd_args: Dict[str, Any],
    test_run: TestRun,
    expected_command: list,
) -> None:
    command = cmd_gen_strategy.generate_test_command({}, cmd_args, test_run)
    assert command == expected_command
