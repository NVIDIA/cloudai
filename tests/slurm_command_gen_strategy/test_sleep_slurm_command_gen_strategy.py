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
from typing import Dict, List
from unittest.mock import Mock

import pytest

from cloudai.core import Test, TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.sleep import SleepCmdArgs, SleepSlurmCommandGenStrategy, SleepTestDefinition


class TestSleepSlurmCommandGenStrategy:
    """Test the SleepSlurmCommandGenStrategy class."""

    @pytest.fixture
    def cmd_gen_strategy(self, slurm_system: SlurmSystem) -> SleepSlurmCommandGenStrategy:
        return SleepSlurmCommandGenStrategy(slurm_system, {})

    @pytest.mark.parametrize(
        "cmd_args_data, expected_command",
        [
            ({"seconds": 60}, ["sleep 60"]),
            ({"seconds": 120}, ["sleep 120"]),
        ],
    )
    def test_generate_test_command(
        self,
        tmp_path: Path,
        cmd_gen_strategy: SleepSlurmCommandGenStrategy,
        cmd_args_data: Dict[str, int],
        expected_command: List[str],
    ) -> None:
        sleep_cmd_args = SleepCmdArgs(seconds=cmd_args_data["seconds"])

        test_def = SleepTestDefinition(
            name="sleep_test",
            description="Simple sleep test",
            test_template_name="default_template",
            cmd_args=sleep_cmd_args,
            extra_env_vars={},
            extra_cmd_args={},
        )

        test_obj = Test(test_definition=test_def, test_template=Mock())

        tr = TestRun(
            test=test_obj,
            num_nodes=1,
            nodes=[],
            output_path=tmp_path / "output",
            name="sleep-job",
        )

        command = cmd_gen_strategy.generate_test_command(
            test_def.extra_env_vars,
            test_def.cmd_args_dict,
            tr,
        )

        assert command == expected_command
