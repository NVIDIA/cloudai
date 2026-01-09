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

from pathlib import Path

import bokeh.plotting as bk
import pandas as pd
import pytest
import toml
from packaging.requirements import Requirement
from packaging.version import Version
from rich.table import Table

from cloudai.core import TestRun, TestScenario
from cloudai.report_generator.comparison_report import ComparisonReport, ComparisonReportConfig
from cloudai.report_generator.groups import GroupedTestRuns, TRGroupItem
from cloudai.systems.slurm import SlurmSystem


class MyComparisonReport(ComparisonReport):
    def extract_data_as_df(self, tr: TestRun) -> pd.DataFrame:
        return pd.DataFrame()

    def create_tables(self, cmp_groups: list[GroupedTestRuns]) -> list[Table]:
        return []

    def create_charts(self, cmp_groups: list[GroupedTestRuns]) -> list[bk.figure]:
        return []


@pytest.fixture
def cmp_report(slurm_system: SlurmSystem) -> MyComparisonReport:
    tc = TestScenario(name="ts", test_runs=[])
    return MyComparisonReport(
        slurm_system, tc, slurm_system.output_path, ComparisonReportConfig(enable=True, group_by=[])
    )


def test_jinja_template_path(cmp_report: MyComparisonReport) -> None:
    full_path = cmp_report.template_path / cmp_report.template_name
    assert full_path.exists()
    assert full_path.is_file()


class TestCreateTable:
    def test_single_data_point(self, cmp_report: MyComparisonReport, nccl_tr: TestRun) -> None:
        table = cmp_report.create_table(
            GroupedTestRuns(
                name="grp_name",
                items=[TRGroupItem(name="item_name", tr=nccl_tr)],
            ),
            [pd.DataFrame({"size": [1, 2, 4], "value": [10, 20, 40]})],
            "title",
            ["size"],
            ["value"],
        )

        assert table.title == "title: grp_name"
        assert len(table.columns) == 2
        assert len(table.rows) == 3

    def test_two_data_points(self, cmp_report: MyComparisonReport, nccl_tr: TestRun) -> None:
        table = cmp_report.create_table(
            GroupedTestRuns(
                name="grp_name",
                items=[
                    TRGroupItem(name="item_name", tr=nccl_tr),
                    TRGroupItem(name="item_name2", tr=nccl_tr),
                ],
            ),
            [
                pd.DataFrame({"size": [1, 2, 4], "value": [10, 20, 40]}),
                pd.DataFrame({"size": [1, 2, 4], "value": [100, 200, 400]}),
            ],
            "title",
            ["size"],
            ["value"],
        )

        assert len(table.columns) == 4
        assert len(table.rows) == 3
        assert list(table.columns[0].cells) == ["1", "2", "4"]
        assert list(table.columns[1].cells) == ["10", "20", "40"]
        assert list(table.columns[2].cells) == ["100", "200", "400"]
        assert list(table.columns[3].cells) == [
            ComparisonReport._format_diff_cell(10, 100),
            ComparisonReport._format_diff_cell(20, 200),
            ComparisonReport._format_diff_cell(40, 400),
        ]

    def test_one_data_point_is_empty(self, cmp_report: MyComparisonReport, nccl_tr: TestRun) -> None:
        table = cmp_report.create_table(
            GroupedTestRuns(
                name="grp_name",
                items=[
                    TRGroupItem(name="item_name", tr=nccl_tr),
                    TRGroupItem(name="item_name2", tr=nccl_tr),
                ],
            ),
            [
                pd.DataFrame({"size": [], "value": []}),
                pd.DataFrame({"size": [1, 2, 4], "value": [10, 20, 40]}),
            ],
            "title",
            ["size"],
            ["value"],
        )

        assert len(table.columns) == 4
        assert table.columns[-1].header == "diff\nvalue"
        assert len(table.rows) == 3
        assert list(table.columns[0].cells) == ["1", "2", "4"]
        assert list(table.columns[1].cells) == ["n/a", "n/a", "n/a"]
        assert list(table.columns[2].cells) == ["10", "20", "40"]
        assert list(table.columns[3].cells) == ["n/a", "n/a", "n/a"]

    def test_two_data_points_with_two_data_columns(self, cmp_report: MyComparisonReport, nccl_tr: TestRun) -> None:
        table = cmp_report.create_table(
            GroupedTestRuns(
                name="grp_name",
                items=[
                    TRGroupItem(name="item_name", tr=nccl_tr),
                    TRGroupItem(name="item_name2", tr=nccl_tr),
                ],
            ),
            [
                pd.DataFrame({"size": [1], "value1": [10], "value2": [5]}),
                pd.DataFrame({"size": [1], "value1": [100], "value2": [50]}),
            ],
            "title",
            ["size"],
            ["value1", "value2"],
        )

        assert len(table.columns) == 7  # 1 info + 2*2 data + 2 diff
        assert len(table.rows) == 1
        assert list(table.columns[0].cells) == ["1"]
        assert list(table.columns[1].cells) == ["10"]
        assert list(table.columns[2].cells) == ["100"]
        assert list(table.columns[3].cells) == [
            ComparisonReport._format_diff_cell(10, 100),
        ]
        assert list(table.columns[4].cells) == ["5"]
        assert list(table.columns[5].cells) == ["50"]
        assert list(table.columns[6].cells) == [
            ComparisonReport._format_diff_cell(5, 50),
        ]

    def test_three_data_points(self, cmp_report: MyComparisonReport, nccl_tr: TestRun) -> None:
        table = cmp_report.create_table(
            GroupedTestRuns(
                name="grp_name",
                items=[
                    TRGroupItem(name="item_name", tr=nccl_tr),
                    TRGroupItem(name="item_name2", tr=nccl_tr),
                    TRGroupItem(name="item_name3", tr=nccl_tr),
                ],
            ),
            [
                pd.DataFrame({"size": [1], "value": [10]}),
                pd.DataFrame({"size": [1], "value": [100]}),
                pd.DataFrame({"size": [1], "value": [1000]}),
            ],
            "title",
            ["size"],
            ["value"],
        )

        assert len(table.columns) == 4  # 1 info + 3*1 data (NO diff)
        assert len(table.rows) == 1
        assert list(table.columns[0].cells) == ["1"]
        assert list(table.columns[1].cells) == ["10"]
        assert list(table.columns[2].cells) == ["100"]
        assert list(table.columns[3].cells) == ["1000"]


def test_create_charts(cmp_report: MyComparisonReport, nccl_tr: TestRun) -> None:
    """This is a sanity test to avoid dumb issues, better coverage might be needed."""
    cmp_report.create_chart(
        GroupedTestRuns(
            name="grp_name",
            items=[
                TRGroupItem(name="item_name", tr=nccl_tr),
                TRGroupItem(name="item_name2", tr=nccl_tr),
            ],
        ),
        [
            pd.DataFrame({"size": [1, 2, 4], "value": [10, 20, 40]}),
            pd.DataFrame({"size": [1, 2, 4], "value": [100, 200, 400]}),
        ],
        "title",
        ["size"],
        ["value"],
        "y_axis_label",
    )


def test_bokeh_cdn_version_matches_pyproject():
    bokeh_dep = None
    for dep in toml.load(Path("pyproject.toml"))["project"]["dependencies"]:
        if dep.startswith("bokeh"):
            bokeh_dep = dep
            break

    assert bokeh_dep is not None, "bokeh dependency not found in pyproject.toml"

    req = Requirement(bokeh_dep)
    assert req.specifier, f"No version specifier found in: {bokeh_dep}"

    template_path = Path("src/cloudai/util/nixl_report_template.jinja2")
    template_content = template_path.read_text()

    pyproject_version = Version(f"{req.specifier}".lstrip("~=<>!"))
    ver_str = f"-{pyproject_version.major}.{pyproject_version.minor}.0"

    for line in template_content.splitlines():
        if "cdn.bokeh.org/bokeh/release" not in line:
            continue

        assert ver_str in line, (
            f"Bokeh CDN version ({line}) in template does not match pyproject.toml version ({pyproject_version})."
        )
