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

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cloudai import BaseJob
from cloudai.systems.lsf import LSFSystem


@pytest.fixture
def lsf_system():
    return LSFSystem(name="test_lsf", install_path=Path("/opt/lsf"), output_path=Path("/tmp/lsf"))


def test_parse_bjobs_output(lsf_system: LSFSystem):
    bjobs_output = "job01 user1 queue1 0 0 host01\njob02 user2 queue2 0 0 host02"
    expected_map = {"host01": "user1", "host02": "user2"}
    result = lsf_system.parse_bjobs_output(bjobs_output)
    assert result == expected_map


def test_parse_bhosts_output(lsf_system: LSFSystem):
    bhosts_output = """host01 ok 0 0 0 queue1
host02 closed 0 0 0 queue2"""
    node_user_map = {"host01": "user1", "host02": "user2"}
    lsf_system.parse_bhosts_output(bhosts_output, node_user_map)

    assert len(lsf_system.queues) == 2
    assert lsf_system.queues[0].name == "queue1"
    assert lsf_system.queues[1].name == "queue2"
    assert lsf_system.queues[0].lsf_nodes[0].name == "host01"
    assert lsf_system.queues[0].lsf_nodes[0].user == "user1"
    assert lsf_system.queues[1].lsf_nodes[0].name == "host02"
    assert lsf_system.queues[1].lsf_nodes[0].user == "user2"


@patch("cloudai.systems.LSFSystem.fetch_command_output")
def test_update(mock_fetch_command_output: Mock, lsf_system: LSFSystem):
    mock_fetch_command_output.side_effect = [
        ("host01 ok 0 0 0 queue1\nhost02 closed 0 0 0 queue2", ""),
        ("job01 user1 queue1 0 0 host01\njob02 user2 queue2 0 0 host02", ""),
    ]
    lsf_system.update()

    assert len(lsf_system.queues) == 2
    assert lsf_system.queues[0].name == "queue1"
    assert lsf_system.queues[1].name == "queue2"


@pytest.mark.parametrize(
    "stdout,expected",
    [
        ("RUN", True),
        ("DONE", False),
        ("EXIT", False),
    ],
)
def test_is_job_running(stdout: str, expected: bool, lsf_system: LSFSystem):
    job = BaseJob(test_run=Mock(), id=1)
    pp = Mock()
    pp.communicate = Mock(return_value=(stdout, ""))
    lsf_system.cmd_shell.execute = Mock(return_value=pp)

    assert lsf_system.is_job_running(job) == expected


@pytest.mark.parametrize(
    "stdout,expected",
    [
        ("DONE", True),
        ("EXIT", True),
        ("RUN", False),
    ],
)
def test_is_job_completed(stdout: str, expected: bool, lsf_system: LSFSystem):
    job = BaseJob(test_run=Mock(), id=1)
    pp = Mock()
    pp.communicate = Mock(return_value=(stdout, ""))
    lsf_system.cmd_shell.execute = Mock(return_value=pp)

    assert lsf_system.is_job_completed(job) == expected


@patch("cloudai.systems.LSFSystem.fetch_command_output")
def test_fetch_command_output(mock_fetch_command_output: Mock, lsf_system: LSFSystem):
    mock_fetch_command_output.return_value = ("output", "error")
    stdout, stderr = lsf_system.fetch_command_output("some_command")
    assert stdout == "output"
    assert stderr == "error"
