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

from typing import Any, Dict, List, Union, cast

from cloudai import TestRun
from cloudai.systems.lsf.strategy import LSFCommandGenStrategy

from .lsf_command_launcher import LSFCommandLauncherTestDefinition


class LSFCommandGenStrategy(LSFCommandGenStrategy):
    """Command generation strategy for generic LSF Command Launcher tests."""

    def generate_test_command(
        self, env_vars: Dict[str, Union[str, List[str]]], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> list[str]:
        tdef: LSFCommandLauncherTestDefinition = cast(LSFCommandLauncherTestDefinition, tr.test.test_definition)
        command_parts: list[str] = [tdef.cmd_args.cmd]
        if tr.test.extra_cmd_args:
            command_parts.append(tr.test.extra_cmd_args)

        return ["bash", "-c", f'"{" ".join(command_parts)}"']

