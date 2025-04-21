# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import Mock

import pytest

from cloudai._core.test import Test
from cloudai._core.test_scenario import TestRun
from cloudai.systems import SlurmSystem
from cloudai.workloads.nim import NimCmdArgs, NimSlurmCommandGenStrategy, NimTestDefinition


@pytest.fixture
def strategy(slurm_system: SlurmSystem) -> NimSlurmCommandGenStrategy:
    return NimSlurmCommandGenStrategy(slurm_system, {})


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {
                "served_model_name": "model",
                "leader_ip": "10.0.0.1",
                "tokenizer": "tok",
            },
            [
                "genai-perf",
                "profile",
                "-m",
                "model",
                "--endpoint-type",
                "chat",
                "--service-kind",
                "openai",
                "--streaming",
                "-u",
                "10.0.0.1:8000",
                "--num-prompts",
                "20",
                "--synthetic-input-tokens-mean",
                "128",
                "--synthetic-input-tokens-stddev",
                "0",
                "--concurrency",
                "1",
                "--output-tokens-mean",
                "128",
                "--extra-inputs",
                "max_tokens:128",
                "--extra-inputs",
                "min_tokens:128",
                "--extra-inputs",
                "ignore_eos:true",
                "--artifact-dir",
                "/cloudai_run_results",
                "--tokenizer",
                "tok",
                "--",
                "-v",
                "--max-threads=1",
                "--request-count=20",
            ],
        )
    ],
)
def test_default_command(strategy, kwargs, expected):
    args = NimCmdArgs(**kwargs)
    td = NimTestDefinition(name="nim", description="", test_template_name="", cmd_args=args)
    tr = TestRun(name="run", test=Test(test_definition=td, test_template=Mock()), nodes=[], num_nodes=1)
    assert strategy.generate_test_command({}, {}, tr) == expected


def test_disable_streaming(strategy):
    args = NimCmdArgs(
        served_model_name="m",
        leader_ip="1.2.3.2",
        tokenizer="tk",
        streaming=False,
    )
    td = NimTestDefinition(name="nim", description="", test_template_name="", cmd_args=args)
    tr = TestRun(name="run", test=Test(test_definition=td, test_template=Mock()), nodes=[], num_nodes=1)
    cmd = strategy.generate_test_command({}, {}, tr)
    assert "--streaming" not in cmd


def test_port_override(strategy):
    args = NimCmdArgs(
        served_model_name="m",
        leader_ip="127.0.0.1",
        port=9001,
        tokenizer="tk",
    )
    td = NimTestDefinition(name="nim", description="", test_template_name="", cmd_args=args)
    tr = TestRun(name="run", test=Test(test_definition=td, test_template=Mock()), nodes=[], num_nodes=1)
    cmd = strategy.generate_test_command({}, {}, tr)
    assert "-u" in cmd
    assert "127.0.0.1:9001" in cmd
