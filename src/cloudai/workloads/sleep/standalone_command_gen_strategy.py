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

from typing import cast

import toml

from cloudai.core import CommandGenStrategy
from cloudai.models.scenario import TestRunDetails

from .sleep import SleepCmdArgs, SleepTestDefinition


class SleepStandaloneCommandGenStrategy(CommandGenStrategy):
    """Command generation strategy for the Sleep test on standalone systems."""

    def store_test_run(self) -> None:
        tdef: SleepTestDefinition = cast(SleepTestDefinition, self.test_run.test)
        sec = tdef.cmd_args.seconds
        test_cmd = f"sleep {sec}"
        self.test_run.output_path.mkdir(parents=True, exist_ok=True)
        with (self.test_run.output_path / self.TEST_RUN_DUMP_FILE_NAME).open("w", encoding="utf-8") as f:
            trd = TestRunDetails.from_test_run(self.test_run, test_cmd=test_cmd, full_cmd=test_cmd)
            toml.dump(trd.model_dump(), f)

    def gen_exec_command(self) -> str:
        self.store_test_run()
        tdef: SleepTestDefinition = cast(SleepTestDefinition, self.test_run.test)
        tdef_cmd_args: SleepCmdArgs = tdef.cmd_args
        sec = tdef_cmd_args.seconds
        return f"sleep {sec}"
