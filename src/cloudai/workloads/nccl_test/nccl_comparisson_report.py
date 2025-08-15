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
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import jinja2
import toml
from rich.console import Console
from rich.table import Table

from cloudai.core import Reporter, System, TestRun, TestScenario
from cloudai.models.scenario import ReportConfig
from cloudai.models.workload import TestDefinition
from cloudai.report_generator.util import add_human_readable_sizes
from cloudai.util.lazy_imports import lazy

from .nccl import NCCLTestDefinition
from .performance_report_generation_strategy import extract_nccl_data

if TYPE_CHECKING:
    import bokeh.plotting as bk
    import pandas as pd


@dataclass
class GroupItem:
    name: str
    tr: TestRun


@dataclass
class GroupedfResult:
    name: str
    items: list[GroupItem]


def diff_trs(trs: list[TestRun]) -> dict[str, list[str]]:
    """Acts like .action_space for a DSE TestRun, but for a list of TestRuns."""
    dicts: list[dict] = []
    for tr in trs:
        dicts.append(
            {
                "NUM_NODES": tr.num_nodes,
                **tr.test.test_definition.cmd_args.model_dump(),
                **{f"extra_env_vars.{k}": v for k, v in tr.test.test_definition.extra_env_vars.items()},
            }
        )
    all_keys = set().union(*[d.keys() for d in dicts])

    diff = {}
    for key in all_keys:
        all_values = [d[key] for d in dicts]
        if len(set(all_values)) > 1:
            diff[key] = all_values

    return diff


def group_for_comparison(trs: list[TestRun], group_by: list[str]) -> list[GroupedfResult]:
    def _grp_name() -> str:
        if not group_by:
            return "all-in-one"
        parts = [f"{field}={_get_value(tr.test.test_definition, field)}" for field in group_by]
        return " ".join(parts).replace("extra_env_vars.", "")

    if len(trs) == 1 or not group_by:
        items: list[GroupItem] = []
        diff = diff_trs(trs)
        for idx, _ in enumerate(trs):
            item_name_parts = [f"{field}={vals[idx]}" for field, vals in diff.items()]
            name = " ".join(item_name_parts).replace("extra_env_vars.", "")
            if not diff:
                name = f"{idx}"
            items.append(GroupItem(name=name, tr=trs[idx]))
        return [GroupedfResult(name=_grp_name(), items=items)]

    def _get_value(tdef: TestDefinition, field: str) -> str:
        if field.startswith("extra_env_vars."):
            f_name = field[len("extra_env_vars.") :]
            v = str(tdef.extra_env_vars.get(f_name))
        else:
            v = getattr(tdef.cmd_args, field)

        return v

    groups: list[list[TestRun]] = []
    for tr in trs:
        for group in groups:
            matched = all(
                _get_value(tr.test.test_definition, field) == _get_value(group[0].test.test_definition, field)
                for field in group_by
            )

            if matched:
                group.append(tr)
                break
        else:  # runs only if no break happened
            groups.append([tr])

    res: list[GroupedfResult] = []
    for grp_idx, group in enumerate(groups):
        items: list[GroupItem] = []
        diff = diff_trs(group)
        for idx, tr in enumerate(group):
            item_name_parts = [f"{field}={vals[idx]}" for field, vals in diff.items()]
            name = " ".join(item_name_parts).replace("extra_env_vars.", "")
            if not diff:
                name = f"{grp_idx}"
            items.append(GroupItem(name=name, tr=tr))
        res.append(GroupedfResult(name=_grp_name(), items=items))
    return res


class NcclComparissonReport(Reporter):
    """Comparisson report for NCCL Test."""

    INFO_COLUMNS = ("Size (B)", "Count", "Type", "Redop")
    LATENCY_DATA_COLUMNS = ("Time (us) Out-of-place", "Time (us) In-place")
    BANDWIDTH_DATA_COLUMNS = ("Busbw (GB/s) Out-of-place", "Busbw (GB/s) In-place")

    def __init__(self, system: System, test_scenario: TestScenario, results_root: Path, config: ReportConfig) -> None:
        super().__init__(system, test_scenario, results_root, config)
        self.group_by: list[str] = [
            # "extra_env_vars.NCCL_TESTS_SPLIT_MASK",
            "subtest_name",
        ]

    def generate(self) -> None:
        self.load_test_runs()
        if not self.trs:
            logging.debug(f"No NCCL results found, skipping {self.__class__.__name__} report generation.")
            return

        console = Console(record=True)
        cmp_groups = group_for_comparison(self.trs, self.group_by)

        for group in cmp_groups:
            table = self.create_table(
                group, title="Latecy", info_columns=self.INFO_COLUMNS, data_columns=self.LATENCY_DATA_COLUMNS
            )
            console.print(table)
            console.print()

            table = self.create_table(
                group, title="Bandwidth", info_columns=self.INFO_COLUMNS, data_columns=self.BANDWIDTH_DATA_COLUMNS
            )
            console.print(table)
            console.print()

        bokeh_script, bokeh_div = self.get_bokeh_html()

        template = jinja2.Environment(
            loader=jinja2.FileSystemLoader(Path(__file__).parent.parent.parent / "util")
        ).get_template("nixl_report_template.jinja2")
        html_content = template.render(
            title=f"{self.test_scenario.name} NCCL Bench Report",
            bokeh_script=bokeh_script,
            bokeh_div=bokeh_div,
            rich_html=console.export_html(),
        )

        html_file = self.results_root / "nccl_comparisson.html"
        with open(html_file, "w") as f:
            f.write(html_content)

        logging.info(f"NCCL comparisson report created: {html_file}")

    def create_table(
        self, group: GroupedfResult, title: str, info_columns: Iterable[str], data_columns: Iterable[str]
    ) -> Table:
        dfs = [self._extract_data(item.tr) for item in group.items]

        style_cycle = cycle(["green", "cyan", "magenta", "blue", "yellow"])

        table = Table(title=f"{title}: {group.name}", title_justify="left", expand=True)
        for col in info_columns:
            table.add_column(col)
        for item in group.items:
            style = next(style_cycle)
            for col in data_columns:
                table.add_column(f"{item.name}\n{col}", overflow="fold", style=style, header_style=style)

        for row_idx in range(len(dfs[0][info_columns[0]])):
            data = []
            for df in dfs:
                data.extend([str(df[col][row_idx]) for col in data_columns])

            table.add_row(*[str(dfs[0][col][row_idx]) for col in info_columns], *data)

        return table

    def _extract_data(self, tr: TestRun) -> pd.DataFrame:
        parsed_data_rows, gpu_type, num_devices_per_node, num_ranks = extract_nccl_data(tr.output_path / "stdout.txt")
        if not parsed_data_rows:
            return lazy.pd.DataFrame()

        df: pd.DataFrame = lazy.pd.DataFrame(
            parsed_data_rows,
            columns=[
                "Size (B)",
                "Count",
                "Type",
                "Redop",
                "Root",
                "Time (us) Out-of-place",
                "Algbw (GB/s) Out-of-place",
                "Busbw (GB/s) Out-of-place",
                "#Wrong Out-of-place",
                "Time (us) In-place",
                "Algbw (GB/s) In-place",
                "Busbw (GB/s) In-place",
                "#Wrong In-place",
            ],
        )

        df["GPU Type"] = gpu_type
        df["Devices per Node"] = num_devices_per_node
        df["Ranks"] = num_ranks

        df["Size (B)"] = df["Size (B)"].astype(int)
        df["Time (us) Out-of-place"] = df["Time (us) Out-of-place"].astype(float).round(2)
        df["Time (us) In-place"] = df["Time (us) In-place"].astype(float).round(2)
        df["Algbw (GB/s) Out-of-place"] = df["Algbw (GB/s) Out-of-place"].astype(float)
        df["Busbw (GB/s) Out-of-place"] = df["Busbw (GB/s) Out-of-place"].astype(float)
        df["Algbw (GB/s) In-place"] = df["Algbw (GB/s) In-place"].astype(float)
        df["Busbw (GB/s) In-place"] = df["Busbw (GB/s) In-place"].astype(float)

        df = add_human_readable_sizes(df, "Size (B)", "Size Human-readable")

        return df

    def get_bokeh_html(self) -> tuple[str, str]:
        cmp_groups = group_for_comparison(self.trs, self.group_by)
        charts: list[bk.figure] = []
        for group in cmp_groups:
            if chart := self.create_chart(group, "Latecy", self.INFO_COLUMNS, self.LATENCY_DATA_COLUMNS, "Time (us)"):
                charts.append(chart)
            if chart := self.create_chart(
                group, "Bandwidth", self.INFO_COLUMNS, self.BANDWIDTH_DATA_COLUMNS, "Busbw (GB/s)"
            ):
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

    def create_chart(
        self,
        group: GroupedfResult,
        title: str,
        info_columns: Iterable[str],
        data_columns: Iterable[str],
        y_axis_label: str,
    ) -> bk.figure | None:
        dfs = [self._extract_data(item.tr) for item in group.items]

        style_cycle = cycle(["green", "cyan", "magenta", "blue", "yellow"])

        p = lazy.bokeh_plotting.figure(
            title=f"{title}: {group.name}",
            x_axis_label=info_columns[0],
            y_axis_label=y_axis_label,
            width=800,
            height=500,
            tools="pan,box_zoom,wheel_zoom,reset,save",
            active_drag="pan",
            active_scroll="wheel_zoom",
            x_axis_type="log",
        )

        hover = lazy.bokeh_models.HoverTool(tooltips=[("X", "@x"), ("Y", "@y"), ("Segment Type", "@segment_type")])
        p.add_tools(hover)

        for df, name in zip(dfs, [item.name for item in group.items], strict=True):
            for col in data_columns:
                source = lazy.bokeh_models.ColumnDataSource(
                    data={
                        "x": df[info_columns[0]].tolist(),
                        "y": df[col].tolist(),
                        "segment_type": [col] * len(df),
                    }
                )

                color = next(style_cycle)
                p.line("x", "y", source=source, line_color=color, line_width=2, legend_label=f"{name} {col}")
                p.scatter("x", "y", source=source, fill_color=color, size=8, legend_label=f"{name} {col}")

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        y_max = max(df[col].max() for df in dfs for col in data_columns)
        y_min = min(df[col].min() for df in dfs for col in data_columns)
        p.y_range = lazy.bokeh_models.Range1d(start=y_min * -1 * y_max * 0.01, end=y_max * 1.1)

        return p
