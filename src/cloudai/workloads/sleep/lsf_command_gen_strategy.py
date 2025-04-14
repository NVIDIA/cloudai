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

from typing import Dict, List, Union, cast

from cloudai import TestRun
from cloudai.systems.lsf.strategy import LSFCommandGenStrategy

from .sleep import SleepCmdArgs, SleepTestDefinition


class SleepLSFCommandGenStrategy(LSFCommandGenStrategy):
    """Command generation strategy for Sleep on LSF systems."""

    def _container_mounts(self, tr: TestRun) -> list[str]:
        return []

    def generate_test_command(
        self, env_vars: Dict[str, Union[str, List[str]]], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        tdef: SleepTestDefinition = cast(SleepTestDefinition, tr.test.test_definition)
        tdef_cmd_args: SleepCmdArgs = tdef.cmd_args
        return [f"sleep {tdef_cmd_args.seconds}"]
