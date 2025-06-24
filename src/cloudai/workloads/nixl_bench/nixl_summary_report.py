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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jinja2
import toml
from rich.console import Console
from rich.table import Table

from cloudai.core import Reporter, System, TestScenario
from cloudai.models.scenario import ReportConfig
from cloudai.util.lazy_imports import lazy

from .nixl_bench import NIXLBenchTestDefinition

if TYPE_CHECKING:
    import bokeh.plotting as bk
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
        self.metric2col = {
            "avg_lat": "Avg. Latency (us)",
            "bw_gb_sec": "Bandwidth (GB/sec)",
        }
        self.report_configs = [
            ("READ", "bw_gb_sec"),
            ("WRITE", "bw_gb_sec"),
            ("READ", "avg_lat"),
            ("WRITE", "avg_lat"),
        ]

    def generate(self) -> None:
        self.load_tdef_with_results()

        console = Console(record=True)
        for op_type, metric in self.report_configs:
            table = self.create_table(op_type, metric)
            console.print(table)
            console.print()

        bokeh_script, bokeh_div = self.get_bokeh_html()

        template = jinja2.Environment(
            loader=jinja2.FileSystemLoader(Path(__file__).parent.parent.parent / "util")
        ).get_template("nixl_report_template.jinja2")
        html_content = template.render(
            title=f"{self.test_scenario.name} NIXL Bench Report",
            bokeh_script=bokeh_script,
            bokeh_div=bokeh_div,
            rich_html=console.export_html(),
        )

        html_file = self.results_root / "nixl_summary.html"
        with open(html_file, "w") as f:
            f.write(html_content)

        logging.info(f"NIXL summary report created: {html_file}")

    def load_tdef_with_results(self) -> None:
        super().load_test_runs()
        self.trs = [tr for tr in self.trs if isinstance(tr.test.test_definition, NIXLBenchTestDefinition)]

        for tr in self.trs:
            tr_file = toml.load(tr.output_path / "test-run.toml")
            tdef = NIXLBenchTestDefinition.model_validate(tr_file["test_definition"])
            self.tdef_res.append(TdefResult(tdef, lazy.pd.read_csv(tr.output_path / "nixlbench.csv")))

    def create_table(self, op_type: str, metric: str) -> Table:
        df = self.construct_df(op_type, metric)
        table = Table(title=f"{self.test_scenario.name}: {op_type} {self.metric2col[metric]}", title_justify="left")
        for col in df.columns:
            table.add_column(col, justify="right", style="cyan")

        for _, row in df.iterrows():
            block_size = row["block_size"].astype(int)
            batch_size = row["batch_size"].astype(int)
            table.add_row(str(block_size), str(batch_size), *[str(x) for x in row.values[2:]])
        return table

    def get_bokeh_html(self) -> tuple[str, str]:
        charts: list[bk.figure] = []
        for op_type, metric in self.report_configs:
            if chart := self.create_chart(op_type, metric):
                charts.append(chart)

        # layout with 2 charts per row
        rows = []
        for i in range(0, len(charts), 2):
            if i + 1 < len(charts):
                rows.append(lazy.bokeh_layouts.row(charts[i], charts[i + 1]))
            else:
                rows.append(lazy.bokeh_layouts.row(charts[i]))
        layout = lazy.bokeh_layouts.column(*rows, name="charts_layout")

        bokeh_script, bokeh_div = lazy.bokeh_embed.components(layout)
        return bokeh_script, bokeh_div

    def construct_df(self, op_type: str, metric: str) -> pd.DataFrame:
        """
        Construct a `DataFrame` with results for all test runs.

        Block size and Batch size are taken only once assuming they are the same across all test runs.
        `op_type` is used to filter the test runs.
        """
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

    def create_chart(self, op_type: str, metric: str) -> bk.figure | None:
        df = self.construct_df(op_type, metric)
        if df.empty:
            logging.warning(f"Empty DataFrame for {op_type} {metric}")
            return None

        numeric_cols = [col for col in df.columns if col not in ["block_size", "batch_size"]]
        grouped_df = df.groupby("block_size")[numeric_cols].mean()
        grouped_df = grouped_df.reset_index()

        colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
        y_columns = [(col, colors[i % len(colors)]) for i, col in enumerate(numeric_cols)]

        p = lazy.bokeh_plotting.figure(
            title=f"{op_type} {self.metric2col[metric]} vs Block Size",
            x_axis_label="Block Size",
            y_axis_label=self.metric2col[metric],
            width=800,
            height=500,
            tools="pan,box_zoom,wheel_zoom,reset,save",
            active_drag="pan",
            active_scroll="wheel_zoom",
            x_axis_type="log",
        )

        hover = lazy.bokeh_models.HoverTool(
            tooltips=[("Block Size", "@x"), ("Value", "@y"), ("Segment Type", "@segment_type")]
        )
        p.add_tools(hover)

        for col, color in y_columns:
            source = lazy.bokeh_models.ColumnDataSource(
                data={
                    "x": grouped_df["block_size"].tolist(),
                    "y": grouped_df[col].tolist(),
                    "segment_type": [col] * len(grouped_df),
                }
            )

            p.line("x", "y", source=source, line_color=color, line_width=2, legend_label=col)
            p.scatter("x", "y", source=source, fill_color=color, size=8, legend_label=col)

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        y_max = grouped_df[numeric_cols].max().max()
        p.y_range = lazy.bokeh_models.Range1d(start=0.0, end=y_max * 1.1)

        return p
