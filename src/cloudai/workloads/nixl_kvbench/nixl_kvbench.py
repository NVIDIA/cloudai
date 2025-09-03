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

from __future__ import annotations

from pydantic import ValidationInfo, field_validator, model_validator

from cloudai.core import CmdArgs, TestDefinition


class NIXLKVBenchCmdArgs(CmdArgs):
    """Command line arguments for NIXLKVBench."""

    # subtest: Literal["sequential-ct-perftest"]
    with_etcd: bool | None = None
    kvbench_script: str = "/workspace/nixl/benchmark/kvbench/main.py"
    python_executable: str = "/workspace/nixl/.venv/bin/python"
    backend: str | None = None

    @model_validator(mode="after")
    def set_with_etcd(self) -> NIXLKVBenchCmdArgs:
        if self.with_etcd is None and self.backend == "UCX":
            self.with_etcd = True
        return self


class NIXLKVBenchTestDefinition(TestDefinition):
    """Test definition for NIXLKVBench."""

    cmd_args: NIXLKVBenchCmdArgs

    @property
    def cmd_args_dict(self) -> dict[str, str | list[str]]:
        return self.cmd_args.model_dump(exclude={"kvbench_script", "python_executable", "with_etcd"})
