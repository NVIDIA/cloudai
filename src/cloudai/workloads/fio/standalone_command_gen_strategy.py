# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import shlex
from typing import cast

import toml

from cloudai.core import CommandGenStrategy
from cloudai.models.scenario import TestRunDetails

from .command import build_fio_command
from .fio import FioTestDefinition


class FioStandaloneCommandGenStrategy(CommandGenStrategy):
    """Command generation strategy for fio on standalone systems."""

    def _generate_fio_command(self) -> str:
        tdef = cast(FioTestDefinition, self.test_run.test)
        exports = [f'export {key}="{value}"' for key, value in self.final_env_vars.items()]
        prefix = [f"{'; '.join(exports)};"] if exports else []
        return build_fio_command(tdef.cmd_args, prefix=prefix)

    def store_test_run(self) -> None:
        test_cmd = self._generate_fio_command()
        self.test_run.output_path.mkdir(parents=True, exist_ok=True)
        full_cmd = self.gen_exec_command(store=False)
        with (self.test_run.output_path / self.TEST_RUN_DUMP_FILE_NAME).open("w", encoding="utf-8") as f:
            trd = TestRunDetails.from_test_run(self.test_run, test_cmd=test_cmd, full_cmd=full_cmd)
            toml.dump(trd.model_dump(exclude_none=True), f)

    def gen_exec_command(self, store: bool = True) -> str:
        if store:
            self.store_test_run()
        stdout = self.test_run.output_path / "stdout.txt"
        stderr = self.test_run.output_path / "stderr.txt"
        return f"{self._generate_fio_command()} > {shlex.quote(str(stdout))} 2> {shlex.quote(str(stderr))}"
