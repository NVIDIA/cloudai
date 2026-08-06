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

from typing import Any, cast

import cloudai.metrics
from cloudai.core import JobStatusResult, System, TestRun
from cloudai.util.lazy_imports import lazy
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

    def metric_observations(self, system: System, tr: TestRun) -> list[cloudai.metrics.MetricObservation]:
        del system
        csv_path = tr.output_path / "nixlbench.csv"
        df = lazy.pd.read_csv(csv_path) if csv_path.is_file() else extract_nixlbench_data(tr.output_path / "stdout.txt")
        observations: list[cloudai.metrics.MetricObservation] = []
        for row in df.itertuples(index=False):
            row = cast(Any, row)
            dimensions: cloudai.metrics.MetricDimensions = {
                "operation": str(getattr(self.cmd_args, "op_type", "default")).lower(),
                "size_bytes": int(row.block_size),
                "batch_size": int(row.batch_size),
                "backend": str(getattr(self.cmd_args, "backend", "default")).lower(),
                "source_memory": str(getattr(self.cmd_args, "initiator_seg_type", "default")).lower(),
                "target_memory": str(getattr(self.cmd_args, "target_seg_type", "default")).lower(),
            }
            observations.extend(
                [
                    cloudai.metrics.MetricObservation(
                        cloudai.metrics.LATENCY,
                        float(row.avg_lat),
                        dimensions,
                        x_dimension=cloudai.metrics.SIZE_BYTES.key,
                    ),
                    cloudai.metrics.MetricObservation(
                        cloudai.metrics.BANDWIDTH,
                        float(row.bw_gb_sec),
                        {**dimensions, "bandwidth_basis": "payload"},
                        x_dimension=cloudai.metrics.SIZE_BYTES.key,
                    ),
                ]
            )
        return observations
