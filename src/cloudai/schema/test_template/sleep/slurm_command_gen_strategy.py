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

from typing import Dict, List

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy


class SleepSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for Sleep on Slurm systems."""

    def gen_exec_command(self, tr: TestRun) -> str:
        return self._write_sbatch_script("sleep", tr)

    def generate_test_command(
        self, env_vars: Dict[str, str], cmd_args: Dict[str, str], extra_cmd_args: str
    ) -> List[str]:
        return [f'sleep {cmd_args["seconds"]}']
