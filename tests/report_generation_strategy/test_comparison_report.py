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

import dataclasses
from pathlib import Path

import pandas as pd
import pytest
import toml
from packaging.requirements import Requirement
from packaging.version import Version

import cloudai.metrics
from cloudai.core import TestRun, TestScenario
from cloudai.report_generator.comparison_report import (
    ComparisonReport,
    ComparisonReportConfig,
    ComparisonSection,
    MetricColumn,
)
from cloudai.report_generator.groups import GroupedTestRuns, TRGroupItem
from cloudai.systems.slurm import SlurmSystem


class MyComparisonReport(ComparisonReport):
    def extract_data_as_df(self, tr: TestRun) -> pd.DataFrame:
        return pd.DataFrame()

    def build_sections(self, cmp_groups: list[GroupedTestRuns]) -> list[ComparisonSection]:
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


def test_jinja_template_path_v2(cmp_report: MyComparisonReport) -> None:
    full_path = cmp_report.template_path / cmp_report.template_name_v2
    assert full_path.exists()
    assert full_path.is_file()


def test_payload_uses_compact_labels_and_structured_differences_v2(
    cmp_report: MyComparisonReport, nccl_tr: TestRun
) -> None:
    long_image = "nvcr.io/example/" + ("very-long-image-name-" * 10) + ":latest"
    item = TRGroupItem(
        name=f"docker_image_url={long_image}",
        tr=nccl_tr,
        compact_name="case-a",
        differences={
            "docker_image_url": long_image,
            "prefill": {"gpu_ids": ["0", "1"], "tensor_parallel_size": 2},
        },
    )
    section = ComparisonSection(
        group=GroupedTestRuns(name="all-in-one", items=[item]),
        dfs=[pd.DataFrame({"size": [1], "value": [10]})],
        title="Throughput",
        info_columns=["size"],
        data_columns=["value"],
        y_axis_label="Requests/s",
    )

    chart = cmp_report._build_chart_v2(section, 0)
    table = cmp_report._build_table_v2(section)
    section_payload = cmp_report._build_sections_v2([section])[0]

    assert chart["datasets"][0]["label"] == "case-a"
    assert "fullLabel" not in chart["datasets"][0]
    assert "borderColor" not in chart["datasets"][0]
    assert table["data_headers"][0]["name"] == "case-a"
    assert table["data_headers"][0]["differences_yaml"] == (
        f"docker_image_url: {long_image}\nprefill:\n  gpu_ids:\n    - '0'\n    - '1'\n  tensor_parallel_size: 2"
    )
    assert section_payload["case_details"][0]["differences_yaml"] == table["data_headers"][0]["differences_yaml"]


def test_indexed_category_axis_uses_display_labels_v2(cmp_report: MyComparisonReport, nccl_tr: TestRun) -> None:
    section = ComparisonSection(
        group=GroupedTestRuns(name="all-in-one", items=[TRGroupItem(name="case-a", tr=nccl_tr)]),
        dfs=[pd.DataFrame({"size": [256, 1024, 4096], "size_label": ["256B", "1KB", "4KB"], "value": [1, 2, 3]})],
        title="Latency",
        info_columns=["size"],
        data_columns=["value"],
        y_axis_label="Time (us)",
        x_axis_type="indexed_category",
        x_axis_column="size_label",
        x_axis_label="Message size",
    )

    chart = cmp_report._build_chart_v2(section, 0)

    assert chart["x_axis_type"] == "indexed_category"
    assert chart["x_axis_label"] == "Message size"
    assert chart["labels"] == ["256B", "1KB", "4KB"]
    assert chart["datasets"][0]["data"] == [1.0, 2.0, 3.0]


def test_auto_y_axis_is_in_payload_v2(cmp_report: MyComparisonReport, nccl_tr: TestRun) -> None:
    section = ComparisonSection(
        group=GroupedTestRuns(name="all-in-one", items=[TRGroupItem(name="case-a", tr=nccl_tr)]),
        dfs=[pd.DataFrame({"metric": ["TTFT", "TPOT"], "value": [3500, 3.5]})],
        title="Latency",
        info_columns=["metric"],
        data_columns=["value"],
        y_axis_label="Latency (ms)",
        chart_type="bar",
        y_axis_type="auto",
    )

    chart = cmp_report._build_chart_v2(section, 0)

    assert chart["y_axis_type"] == "auto"


def test_metric_column_automatically_injects_sol_into_table_and_chart_v2(
    cmp_report: MyComparisonReport, nccl_tr: TestRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = cloudai.metrics.MetricObservation(
        cloudai.metrics.TRANSFER_BANDWIDTH,
        80,
        cloudai.metrics.TransferCoordinates(payload_size_bytes=1024),
    )
    assessment = cloudai.metrics.assess_observation(
        observation,
        cloudai.metrics.parse_sol_spec({"transfer_bandwidth": [{"value": 100}]}),
    )
    monkeypatch.setattr(cmp_report, "_assessments", lambda tr: [assessment])
    section = ComparisonSection(
        group=GroupedTestRuns(name="all-in-one", items=[TRGroupItem(name="case-a", tr=nccl_tr)]),
        dfs=[pd.DataFrame({"size": [1024], "bandwidth": [80]})],
        title="Bandwidth",
        info_columns=["size"],
        data_columns=["bandwidth"],
        y_axis_label="GB/s",
        x_axis_type="linear",
        metric_columns={
            "bandwidth": MetricColumn(
                cloudai.metrics.TRANSFER_BANDWIDTH,
                coordinate_columns={"payload_size_bytes": "size"},
            )
        },
    )

    table = cmp_report._build_table_v2(section)
    chart = cmp_report._build_chart_v2(section, 0)

    assert [header["name"] for header in table["data_headers"]] == ["case-a", "case-a · SOL", "case-a · % SOL"]
    assert table["rows"][0]["data_cells"] == ["80", "100.0", "80.0%"]
    assert chart["datasets"][1]["label"] == "SOL"
    assert chart["datasets"][1]["data"] == [{"x": 1024.0, "y": 100.0}]
    assert chart["datasets"][1]["is_sol"] is True
    assert chart["sol_color"] == "#741D9D"


def test_chart_v2_renders_one_shared_sol_curve(
    cmp_report: MyComparisonReport, nccl_tr: TestRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    second_tr = dataclasses.replace(nccl_tr, name="case-b")
    observation = cloudai.metrics.MetricObservation(
        cloudai.metrics.TRANSFER_BANDWIDTH,
        80,
        cloudai.metrics.TransferCoordinates(payload_size_bytes=1024),
    )
    assessment = cloudai.metrics.assess_observation(
        observation,
        cloudai.metrics.parse_sol_spec({"transfer_bandwidth": [{"value": 100}]}),
    )
    monkeypatch.setattr(cmp_report, "_assessments", lambda tr: [assessment])
    section = ComparisonSection(
        group=GroupedTestRuns(
            name="all-in-one",
            items=[TRGroupItem(name="case-a", tr=nccl_tr), TRGroupItem(name="case-b", tr=second_tr)],
        ),
        dfs=[
            pd.DataFrame({"size": [1024], "bandwidth": [80]}),
            pd.DataFrame({"size": [1024], "bandwidth": [90]}),
        ],
        title="Bandwidth",
        info_columns=["size"],
        data_columns=["bandwidth"],
        y_axis_label="GB/s",
        x_axis_type="linear",
        metric_columns={
            "bandwidth": MetricColumn(
                cloudai.metrics.TRANSFER_BANDWIDTH,
                coordinate_columns={"payload_size_bytes": "size"},
            )
        },
    )

    chart = cmp_report._build_chart_v2(section, 0)

    sol_datasets = [dataset for dataset in chart["datasets"] if dataset.get("is_sol")]
    assert sol_datasets == [{"label": "SOL", "data": [{"x": 1024.0, "y": 100.0}], "is_sol": True}]


def test_chart_v2_omits_sol_when_compared_runs_disagree(
    cmp_report: MyComparisonReport, nccl_tr: TestRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    second_tr = dataclasses.replace(nccl_tr, name="case-b")
    observation = cloudai.metrics.MetricObservation(
        cloudai.metrics.TRANSFER_BANDWIDTH,
        80,
        cloudai.metrics.TransferCoordinates(payload_size_bytes=1024),
    )
    assessments = {
        id(nccl_tr): cloudai.metrics.assess_observation(
            observation,
            cloudai.metrics.parse_sol_spec({"transfer_bandwidth": [{"value": 100}]}),
        ),
        id(second_tr): cloudai.metrics.assess_observation(
            observation,
            cloudai.metrics.parse_sol_spec({"transfer_bandwidth": [{"value": 120}]}),
        ),
    }
    monkeypatch.setattr(cmp_report, "_assessments", lambda tr: [assessments[id(tr)]])
    section = ComparisonSection(
        group=GroupedTestRuns(
            name="all-in-one",
            items=[TRGroupItem(name="case-a", tr=nccl_tr), TRGroupItem(name="case-b", tr=second_tr)],
        ),
        dfs=[
            pd.DataFrame({"size": [1024], "bandwidth": [80]}),
            pd.DataFrame({"size": [1024], "bandwidth": [90]}),
        ],
        title="Bandwidth",
        info_columns=["size"],
        data_columns=["bandwidth"],
        y_axis_label="GB/s",
        x_axis_type="linear",
        metric_columns={
            "bandwidth": MetricColumn(
                cloudai.metrics.TRANSFER_BANDWIDTH,
                coordinate_columns={"payload_size_bytes": "size"},
            )
        },
    )

    chart = cmp_report._build_chart_v2(section, 0)

    assert not any(dataset.get("is_sol") for dataset in chart["datasets"])


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
