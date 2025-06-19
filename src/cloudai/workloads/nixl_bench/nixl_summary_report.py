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

    def _construct_df(self, op_type: str, metric: str) -> pd.DataFrame:
        final_df = lazy.pd.DataFrame()

        for tdef_res in self.tdef_res:
            if tdef_res.tdef.cmd_args_dict.get("op_type", "unset") != op_type:
                continue
            if final_df.empty:
                final_df["block_size"] = tdef_res.results["block_size"].astype(int)
                final_df["batch_size"] = tdef_res.results["batch_size"].astype(int)

            col_name = (
                f"{tdef_res.tdef.cmd_args_dict.get('initiator_seg_type', 'unset')}->"
                f"{tdef_res.tdef.cmd_args_dict.get('target_seg_type', 'unset')}"
            )
            final_df[col_name] = tdef_res.results[metric].astype(float)

        return final_df

    def create_table(self, op_type: str, metric: str) -> Table:
        metric2col = {
            "avg_lat": "Avg. Latency (us)",
            "bw_gb_sec": "Bandwidth (GB/sec)",
        }

        df = self._construct_df(op_type, metric)
        table = Table(title=f"{self.test_scenario.name}: {op_type} {metric2col[metric]}")
        for col in df.columns:
            table.add_column(col, justify="right", style="cyan")

        for _, row in df.iterrows():
            block_size = row["block_size"].astype(int)
            batch_size = row["batch_size"].astype(int)
            table.add_row(str(block_size), str(batch_size), *[str(x) for x in row.values[2:]])
        return table

    def generate(self) -> None:
        self.load_tdef_res()

        console = Console()
        for op_type in ["READ", "WRITE"]:
            for metric in ["avg_lat", "bw_gb_sec"]:
                table = self.create_table(op_type, metric)
                console.print(table)
