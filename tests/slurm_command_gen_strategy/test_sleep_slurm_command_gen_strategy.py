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

from typing import Dict, List, Union
from unittest.mock import Mock

import pytest

from cloudai.systems import SlurmSystem
from cloudai.workloads.sleep import SleepSlurmCommandGenStrategy


class TestSleepSlurmCommandGenStrategy:
    @pytest.fixture
    def cmd_gen_strategy(self, slurm_system: SlurmSystem) -> SleepSlurmCommandGenStrategy:
        return SleepSlurmCommandGenStrategy(slurm_system, {})

    @pytest.mark.parametrize(
        "cmd_args, expected_command",
        [
            ({"seconds": "60"}, ["sleep 60"]),
            ({"seconds": "120"}, ["sleep 120"]),
        ],
    )
    def test_generate_test_command(
        self,
        cmd_gen_strategy: SleepSlurmCommandGenStrategy,
        cmd_args: Dict[str, Union[str, List[str]]],
        expected_command: List[str],
    ) -> None:
        env_vars = {}
        command = cmd_gen_strategy.generate_test_command(env_vars, cmd_args, Mock())
        assert command == expected_command
