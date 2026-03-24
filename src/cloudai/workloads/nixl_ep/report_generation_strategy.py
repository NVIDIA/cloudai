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

import json
from pathlib import Path
from typing import ClassVar

from rich.console import Console
from rich.table import Table

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy

from .log_parsing import parse_nixl_ep_bandwidth_samples, parse_nixl_ep_completed_phases
from .nixl_ep import GENERATED_PLAN_FILE_NAME


class NixlEPReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from NIXL EP benchmark directories."""

    metrics: ClassVar[list[str]] = ["default"]

    def _node_logs(self) -> list[Path]:
        return [self.test_run.output_path / f"nixl-ep-node-{i}.log" for i in range(self.test_run.nnodes)]

    def can_handle_directory(self) -> bool:
        return any(parse_nixl_ep_bandwidth_samples(path) for path in self._node_logs())

    def _load_plan(self) -> list[list[int]]:
        plan_path = self.test_run.output_path / GENERATED_PLAN_FILE_NAME
        if not plan_path.is_file():
            return []
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            return plan if isinstance(plan, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def generate_report(self) -> None:
        console = Console()
        node_logs = self._node_logs()
        plan = self._load_plan()
        num_phases = len(plan)

        if not node_logs:
            console.print("[yellow]NIXL EP: no node logs found[/yellow]")
            return

        completed_by_node = {i: parse_nixl_ep_completed_phases(log) for i, log in enumerate(node_logs)}
        samples_by_node = {i: parse_nixl_ep_bandwidth_samples(log) for i, log in enumerate(node_logs)}

        has_combined = any(s.dispatch_combine_bandwidth_gbps is not None for ss in samples_by_node.values() for s in ss)
        has_kineto = any(s.dispatch_bandwidth_gbps is not None for ss in samples_by_node.values() for s in ss)

        passed = sum(1 for p in range(num_phases) if p in completed_by_node.get(0, set()))
        case_id = self.test_run.name
        phases_summary = f"{passed}/{num_phases} phases passed" if num_phases else ""
        title = f"NIXL EP — {case_id}" + (f" — {phases_summary}" if phases_summary else "")

        table = Table(title=title, show_lines=True)
        table.add_column("Node", justify="right")
        table.add_column("Phases", justify="center")
        if has_combined:
            table.add_column("Dispatch+Combine BW (GB/s)", justify="right")
            table.add_column("Avg (µs)", justify="right")
            table.add_column("Min (µs)", justify="right")
            table.add_column("Max (µs)", justify="right")
        if has_kineto:
            table.add_column("Dispatch BW (GB/s)", justify="right")
            table.add_column("Combine BW (GB/s)", justify="right")

        def mean(vals: list[float]) -> float | None:
            return sum(vals) / len(vals) if vals else None

        def fmt(v: float | None) -> str:
            return f"{v:.2f}" if v is not None else "—"

        def phase_cell(completed: set[int]) -> str:
            if not plan:
                return "—"
            parts = []
            for p, ranks in enumerate(plan):
                label = str(ranks)
                parts.append(f"[green]{label}[/green]" if p in completed else f"[red]{label}[/red]")
            return "\n".join(parts)

        for node_idx in range(len(node_logs)):
            completed = completed_by_node.get(node_idx, set())
            samples = samples_by_node.get(node_idx, [])
            phases_str = phase_cell(completed)

            row = [str(node_idx), phases_str]
            if has_combined:
                row += [
                    fmt(mean([s.dispatch_combine_bandwidth_gbps for s in samples if s.dispatch_combine_bandwidth_gbps is not None])),
                    fmt(mean([s.avg_time_us for s in samples if s.avg_time_us is not None])),
                    fmt(mean([s.min_time_us for s in samples if s.min_time_us is not None])),
                    fmt(mean([s.max_time_us for s in samples if s.max_time_us is not None])),
                ]
            if has_kineto:
                row += [
                    fmt(mean([s.dispatch_bandwidth_gbps for s in samples if s.dispatch_bandwidth_gbps is not None])),
                    fmt(mean([s.combine_bandwidth_gbps for s in samples if s.combine_bandwidth_gbps is not None])),
                ]
            table.add_row(*row)

        console.print(table)

    def get_metric(self, metric: str) -> float:
        if metric not in self.metrics:
            return METRIC_ERROR
        samples = [s for path in self._node_logs() for s in parse_nixl_ep_bandwidth_samples(path)]
        bw_values = [
            s.dispatch_combine_bandwidth_gbps
            for s in samples
            if s.dispatch_combine_bandwidth_gbps is not None
        ]
        if not bw_values:
            return METRIC_ERROR
        return sum(bw_values) / len(bw_values)
