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

from pathlib import Path

import pytest
import toml
from cloudai import Registry
from cloudai._core.test_definitions import (
    ChakraReplayTestDefinition,
    NCCLTestDefinition,
    NeMoLauncherTestDefinition,
    SleepTestDefinition,
    UCCTestDefinition,
)

TOML_FILES = list(Path("conf").glob("**/*.toml"))
ALL_TESTS = [t for t in TOML_FILES if "test_template_name" in t.read_text()]
UCC_TESTS = [t for t in ALL_TESTS if 'test_template_name = "UCCTest"' in t.read_text()]
NCCL_TESTS = [t for t in ALL_TESTS if 'test_template_name = "NcclTest"' in t.read_text()]
NEMO_TESTS = [t for t in ALL_TESTS if 'test_template_name = "NeMoLauncher"' in t.read_text()]
CHAKRA_REPLAY_TESTS = [t for t in ALL_TESTS if 'test_template_name = "ChakraReplay"' in t.read_text()]
SLEEP_TESTS = [t for t in ALL_TESTS if 'test_template_name = "Sleep"' in t.read_text()]


@pytest.mark.parametrize("toml_file", TOML_FILES, ids=lambda x: str(x))
def test_toml_files(toml_file: Path):
    """
    Validate the syntax of a .toml file.

    Args:
        toml_file (Path): The path to the .toml file to validate.
    """
    with toml_file.open("r") as f:
        assert toml.load(f) is not None


@pytest.mark.parametrize("toml_file", ALL_TESTS, ids=lambda x: str(x))
def test_all_tests(toml_file: Path):
    with toml_file.open("r") as f:
        toml_dict = toml.load(f)

    registry = Registry()
    template_name = toml_dict["test_template_name"]
    assert template_name in registry.test_definitions_map, f"Unknown test template: {template_name}"


@pytest.mark.parametrize("toml_file", UCC_TESTS, ids=lambda x: str(x))
def test_ucc_tests(toml_file: Path):
    with toml_file.open("r") as f:
        toml_dict = toml.load(f)

    t = UCCTestDefinition(**toml_dict)
    assert t.name == toml_dict["name"]
    assert t.description == toml_dict["description"]
    assert t.test_template_name == toml_dict["test_template_name"]
    assert t.cmd_args.docker_image_url
    assert t.cmd_args.collective
    assert t.cmd_args.b
    assert t.cmd_args.e


@pytest.mark.parametrize("toml_file", NCCL_TESTS, ids=lambda x: str(x))
def test_nccl_tests(toml_file: Path):
    with toml_file.open("r") as f:
        toml_dict = toml.load(f)

    t = NCCLTestDefinition(**toml_dict)
    assert t.name == toml_dict["name"]
    assert t.description == toml_dict["description"]
    assert t.test_template_name == toml_dict["test_template_name"]
    assert t.cmd_args.docker_image_url
    assert t.cmd_args.subtest_name
    assert t.cmd_args.nthreads
    assert t.cmd_args.ngpus
    assert t.cmd_args.minbytes
    assert t.cmd_args.maxbytes
    assert t.cmd_args.stepbytes
    assert t.cmd_args.op
    assert t.cmd_args.datatype
    assert t.cmd_args.root is not None
    assert t.cmd_args.iters
    assert t.cmd_args.warmup_iters
    assert t.cmd_args.agg_iters
    assert t.cmd_args.average
    assert t.cmd_args.parallel_init is not None
    assert t.cmd_args.check is not None
    assert t.cmd_args.blocking is not None
    assert t.cmd_args.cudagraph is not None


@pytest.mark.parametrize("toml_file", NEMO_TESTS, ids=lambda x: str(x))
def test_nemo_tests(toml_file: Path):
    with toml_file.open("r") as f:
        toml_dict = toml.load(f)

    t = NeMoLauncherTestDefinition(**toml_dict)
    assert t.name == toml_dict["name"]
    assert t.description == toml_dict["description"]
    assert t.test_template_name == toml_dict["test_template_name"]
    assert t.cmd_args.docker_image_url
    # TODO: check other fields


@pytest.mark.parametrize("toml_file", CHAKRA_REPLAY_TESTS, ids=lambda x: str(x))
def test_chakra_replay_tests(toml_file: Path):
    with toml_file.open("r") as f:
        toml_dict = toml.load(f)

    t = ChakraReplayTestDefinition(**toml_dict)
    assert t.name == toml_dict["name"]
    assert t.description == toml_dict["description"]
    assert t.test_template_name == toml_dict["test_template_name"]
    assert t.cmd_args.docker_image_url
    assert t.cmd_args.mpi
    assert t.cmd_args.trace_type
    assert t.cmd_args.trace_path
    assert t.cmd_args.backend
    assert t.cmd_args.device


@pytest.mark.parametrize("toml_file", SLEEP_TESTS, ids=lambda x: str(x))
def test_sleep_tests(toml_file: Path):
    with toml_file.open("r") as f:
        toml_dict = toml.load(f)

    t = SleepTestDefinition(**toml_dict)
    assert t.name == toml_dict["name"]
    assert t.description == toml_dict["description"]
    assert t.test_template_name == toml_dict["test_template_name"]
    assert t.cmd_args.seconds
