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

import pytest

from cloudai._core.pinning_strategy import (
    AwsPinningStrategy,
    AzurePinningStrategy,
    NoOpPinningStrategy,
    create_pinning_strategy,
)


def test_noop_pinning_strategy() -> None:
    ps = NoOpPinningStrategy(0, 0)
    flags = ps.get_pinning_flags()
    assert flags == []


@pytest.mark.parametrize(
    "cpus_per_node,num_tasks_per_node,expected",
    [
        (4, 1, ["--cpu-bind=verbose", "--cpus-per-task=4", "--hint=nomultithread"]),
        (12, 1, ["--cpu-bind=verbose", "--cpus-per-task=12", "--hint=nomultithread"]),
        (16, 2, ["--cpu-bind=verbose", "--cpus-per-task=8", "--hint=nomultithread"]),
        (48, 4, ["--cpu-bind=verbose", "--cpus-per-task=12", "--hint=nomultithread"]),
    ],
)
def test_aws_pinning_strategy(cpus_per_node: int, num_tasks_per_node: int, expected: list[str]) -> None:
    ps = AwsPinningStrategy(cpus_per_node, num_tasks_per_node)
    flags = ps.get_pinning_flags()
    assert flags == expected


@pytest.mark.parametrize(
    "total_cores,num_tasks,expected",
    [
        (48, 4, ['--cpu-bind=mask_cpu:"fff,fff000,fff000000,fff000000000"']),
        (
            96,
            8,
            [
                '--cpu-bind=mask_cpu:"fff,fff000,fff000000,fff000000000,fff000000000000,fff000000000000000,fff000000000000000000,fff000000000000000000000"'
            ],
        ),
        (16, 2, ['--cpu-bind=mask_cpu:"ff,ff00"']),
    ],
)
def test_azure_pinning_strategy(total_cores: int, num_tasks: int, expected: list[str]) -> None:
    ps = AzurePinningStrategy(total_cores, num_tasks)
    flags = ps.get_pinning_flags()
    assert flags == expected


@pytest.mark.parametrize(
    "system_type,cpus_per_node,num_tasks_per_node,expected_class,expected_flags",
    [
        ("unknown", 0, 0, NoOpPinningStrategy, []),
        ("dgx", 32, 4, NoOpPinningStrategy, []),
        ("aws", 16, 4, AwsPinningStrategy, ["--cpu-bind=verbose", "--cpus-per-task=4", "--hint=nomultithread"]),
        ("AWS", 8, 2, AwsPinningStrategy, ["--cpu-bind=verbose", "--cpus-per-task=4", "--hint=nomultithread"]),
        ("azure", 48, 4, AzurePinningStrategy, ['--cpu-bind=mask_cpu:"fff,fff000,fff000000,fff000000000"']),
    ],
)
def test_create_pinning_strategy(
    system_type: str,
    cpus_per_node: int,
    num_tasks_per_node: int,
    expected_class: type,
    expected_flags: list[str],
) -> None:
    ps = create_pinning_strategy(system_type, cpus_per_node, num_tasks_per_node)
    assert isinstance(ps, expected_class)
    assert ps.get_pinning_flags() == expected_flags
