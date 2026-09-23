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

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pydantic

import cloudai.metrics
from cloudai.core import JobStatusResult, System, TestRun
from cloudai.util.lazy_imports import lazy
from cloudai.workloads.common.nixl import (
    MANAGED_ETCD_ENDPOINTS,
    NIXLBaseCmdArgs,
    NIXLBaseTestDefinition,
    NIXLExtendedCmdArgs,
    extract_nixlbench_data,
)

if TYPE_CHECKING:
    import pandas as pd


def read_nixlbench_results(output_path: Path) -> pd.DataFrame:
    """Read legacy output or validate and collect every independent task's results."""
    task_path = output_path / "nixlbench"
    if not task_path.is_dir():
        return extract_nixlbench_data(output_path / "stdout.txt")

    try:
        ntasks = int((task_path / "ntasks").read_text().strip())
        if ntasks < 1:
            raise ValueError("Expected a positive NIXLBench task count.")

        frames = []
        expected_measurements = None
        for task_id in range(ntasks):
            status = int((task_path / f"{task_id}.status").read_text().strip())
            if status != 0:
                raise ValueError(f"NIXLBench task {task_id} exited with status {status}.")
            frame = extract_nixlbench_data(task_path / f"{task_id}.stdout").copy()
            if frame.empty:
                raise ValueError(f"NIXLBench data not found for task {task_id}.")
            if not lazy.np.isfinite(frame[["avg_lat", "bw_gb_sec"]].to_numpy()).all():
                raise ValueError(f"NIXLBench task {task_id} contains non-finite measurements.")

            measurements = list(frame[["block_size", "batch_size"]].itertuples(index=False, name=None))
            if len(set(measurements)) != len(measurements):
                raise ValueError(f"NIXLBench task {task_id} contains duplicate measurements.")
            if expected_measurements is not None and set(measurements) != expected_measurements:
                raise ValueError(f"NIXLBench task {task_id} has a different set of measurements.")
            expected_measurements = set(measurements)

            frame["task_id"] = task_id
            frame["hostname"] = (task_path / f"{task_id}.hostname").read_text().strip()
            frames.append(frame)
        return lazy.pd.concat(frames, ignore_index=True)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid NIXLBench results in {task_path}: {exc}") from exc


def aggregate_nixlbench_results(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize matching measurements, retaining mean bandwidth for legacy consumers."""
    if "task_id" not in df.columns:
        return df
    return df.groupby(["block_size", "batch_size"], as_index=False).agg(
        avg_lat=("avg_lat", "mean"),
        bw_gb_sec=("bw_gb_sec", "mean"),
        bw_min_gb_sec=("bw_gb_sec", "min"),
        bw_sum_gb_sec=("bw_gb_sec", "sum"),
        task_count=("task_id", "count"),
    )


class NIXLBenchCmdArgs(NIXLBaseCmdArgs, NIXLExtendedCmdArgs):
    """Command line arguments for a NIXL Bench test."""

    path_to_benchmark: str
    etcd_endpoints: str = MANAGED_ETCD_ENDPOINTS
    runtime_type: Literal["ETCD", "ASIO"] = "ETCD"
    asio_address: str = "$NIXL_ASIO_ADDRESS"
    asio_port: int = pydantic.Field(default=12345, ge=1, le=65535)


class NIXLBenchTestDefinition(NIXLBaseTestDefinition[NIXLBenchCmdArgs]):
    """Test definition for a NIXL Bench test."""

    @property
    def uses_etcd(self) -> bool:
        """Return whether CloudAI should launch ETCD for this benchmark."""
        return self.cmd_args.runtime_type == "ETCD" and self.cmd_args.etcd_endpoints == MANAGED_ETCD_ENDPOINTS

    @property
    def uses_asio(self) -> bool:
        """Return whether this benchmark uses NIXLBench's ASIO runtime."""
        return self.cmd_args.runtime_type == "ASIO"

    @property
    def cmd_args_dict(self) -> dict[str, str | list[str]]:
        cmd_args = self.cmd_args.model_dump(
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
        if self.cmd_args.runtime_type == "ETCD":
            cmd_args.pop("runtime_type")
            cmd_args.pop("asio_address")
            cmd_args.pop("asio_port")
            if not self.cmd_args.etcd_endpoints:
                cmd_args.pop("etcd_endpoints")
        else:
            cmd_args.pop("etcd_endpoints")
        return cmd_args

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        try:
            df = read_nixlbench_results(tr.output_path)
        except ValueError as exc:
            return JobStatusResult(is_successful=False, error_message=str(exc))
        if df.empty:
            return JobStatusResult(is_successful=False, error_message=f"NIXLBench data not found in {tr.output_path}.")

        return JobStatusResult(is_successful=True)

    def metric_observations(self, system: System, tr: TestRun) -> list[cloudai.metrics.MetricObservation]:
        del system
        csv_path = tr.output_path / "nixlbench.csv"
        try:
            if (tr.output_path / "nixlbench").is_dir() or not csv_path.is_file():
                df = aggregate_nixlbench_results(read_nixlbench_results(tr.output_path))
            else:
                df = lazy.pd.read_csv(csv_path)
        except ValueError as exc:
            logging.warning(str(exc))
            return []
        observations: list[cloudai.metrics.MetricObservation] = []
        for row in df.itertuples(index=False):
            row = cast(Any, row)
            dimensions: cloudai.metrics.MetricDimensions = {
                "operation": str(getattr(self.cmd_args, "op_type", None) or "default").lower(),
                "size_bytes": int(row.block_size),
                "batch_size": int(row.batch_size),
                "backend": str(getattr(self.cmd_args, "backend", None) or "default").lower(),
                "source_memory": str(getattr(self.cmd_args, "initiator_seg_type", None) or "default").lower(),
                "target_memory": str(getattr(self.cmd_args, "target_seg_type", None) or "default").lower(),
            }
            observations.extend(
                [
                    cloudai.metrics.MetricObservation(
                        cloudai.metrics.LATENCY,
                        float(row.avg_lat),
                        dimensions,
                    ),
                    cloudai.metrics.MetricObservation(
                        cloudai.metrics.BANDWIDTH,
                        float(row.bw_gb_sec),
                        {**dimensions, "bandwidth_basis": "payload"},
                    ),
                ]
            )
        return observations
