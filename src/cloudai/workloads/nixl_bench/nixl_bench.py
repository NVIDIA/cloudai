# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai.core import JobStatusResult, TestRun
from cloudai.workloads.common.nixl import (
    NIXLBaseCmdArgs,
    NIXLBaseTestDefinition,
    NIXLExtendedCmdArgs,
    extract_nixlbench_data,
)


class NIXLBenchCmdArgs(NIXLBaseCmdArgs, NIXLExtendedCmdArgs):
    """Command line arguments for a NIXL Bench test."""

    path_to_benchmark: str
    etcd_endpoints: str = "http://$NIXL_ETCD_ENDPOINTS"


class NIXLBenchTestDefinition(NIXLBaseTestDefinition[NIXLBenchCmdArgs]):
    """Test definition for a NIXL Bench test."""

    @property
    def cmd_args_dict(self) -> dict[str, str | list[str]]:
        return self.cmd_args.model_dump(
            exclude={
                "docker_image_url",
                "path_to_benchmark",
                "cmd_args",
                "etcd_path",
                "wait_etcd_for",
                "etcd_image_url",
            },
            exclude_none=True,
        )

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        df = extract_nixlbench_data(tr.output_path / "stdout.txt")
        if df.empty:
            return JobStatusResult(is_successful=False, error_message=f"NIXLBench data not found in {tr.output_path}.")

        return JobStatusResult(is_successful=True)
