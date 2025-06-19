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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import toml
from rich.console import Console
from rich.table import Table

from cloudai.core import Reporter, System, TestScenario
from cloudai.models.scenario import ReportConfig
from cloudai.util.lazy_imports import lazy

from .nixl_bench import NIXLBenchTestDefinition

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class TdefResult:
    """Convenience class for storing test definition and dataframe results."""

    tdef: NIXLBenchTestDefinition
    results: pd.DataFrame


class NIXLBenchSummaryReport(Reporter):
    """Summary report for NIXL Bench."""

    def __init__(self, system: System, test_scenario: TestScenario, results_root: Path, config: ReportConfig) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self.tdef_res: list[TdefResult] = []

    def load_tdef_res(self):
        super().load_test_runs()
        self.tdef_res: list[TdefResult] = []
        self.trs = [tr for tr in self.trs if isinstance(tr.test.test_definition, NIXLBenchTestDefinition)]

        for tr in self.trs:
            tr_file = toml.load(tr.output_path / "test-run.toml")
            tdef = NIXLBenchTestDefinition.model_validate(tr_file["test_definition"])
            self.tdef_res.append(TdefResult(tdef, lazy.pd.read_csv(tr.output_path / "nixlbench.csv")))

    def generate(self) -> None:
        self.load_tdef_res()

        for op_type in ["read", "write"]:
            self.generate_table(op_type.upper(), "Avg. Latency (us)")
            self.generate_table(op_type.upper(), "Bandwidth (GB/sec)")

    def generate_table(self, op_type: str, metric: str):
        table = Table(title=f"{self.test_scenario.name}: {op_type}, {metric}")

        table.add_column("Block Size", justify="right", style="cyan")
        table.add_column("Batch Size", justify="right", style="cyan")

        data: dict[str, list] = {}

        for tdef_res in self.tdef_res:
            case_op_type = tdef_res.tdef.cmd_args_dict.get("op_type", "unset")
            if op_type != case_op_type:
                continue

            in_seg_type = tdef_res.tdef.cmd_args_dict.get("initiator_seg_type", "unset")
            target_seg_type = tdef_res.tdef.cmd_args_dict.get("target_seg_type", "unset")
            bname = f"{in_seg_type}->{target_seg_type}"

            table.add_column(f"{bname}")

            metric_field = {"Avg. Latency (us)": "avg_lat", "Bandwidth (GB/sec)": "bw_gb_sec"}[metric]

            for _, row in tdef_res.results.iterrows():
                key = str(row["block_size"].astype(int)) + "--" + str(row["batch_size"].astype(int))
                if key not in data:
                    data[key] = []
                data[key].extend([str(row[metric_field])])

        for k, v in data.items():
            block_size, batch_size = k.split("--")
            table.add_row(block_size, batch_size, *v)

        console = Console()
        console.print(table)
