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


from cloudai import CommandGenStrategy, TestRun


class SleepStandaloneCommandGenStrategy(CommandGenStrategy):
    """
    Command generation strategy for the Sleep test on standalone systems.

    This strategy generates a command to execute a sleep operation with specified duration on standalone systems.
    """

    def gen_exec_command(self, tr: TestRun) -> str:
        self.final_cmd_args = self._override_cmd_args(self.default_cmd_args, tr.test.cmd_args)
        sec = self.final_cmd_args["seconds"]
        return f"sleep {sec}"
