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

import copy
from pathlib import Path

import pandas as pd
import pytest
import toml
from packaging.requirements import Requirement
from packaging.version import Version

from cloudai.core import TestRun, TestScenario
from cloudai.report_generator.comparison_report import ComparisonReport, ComparisonReportConfig, ComparisonSection
from cloudai.report_generator.groups import GroupedTestRuns, TRGroupItem
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.ai_dynamo import AIDynamoComparisonReport
from cloudai.workloads.nccl_test import NcclComparisonReport
from cloudai.workloads.nixl_bench.nixl_summary_report import NIXLBenchComparisonReport
from cloudai.workloads.nixl_ep import NixlEPComparisonReport
from cloudai.workloads.osu_bench import OSUBenchComparisonReport
from cloudai.workloads.sglang import SGLangComparisonReport
from cloudai.workloads.vllm import VLLMComparisonReport


class MyComparisonReport(ComparisonReport):
    def extract_data_as_df(self, tr: TestRun) -> pd.DataFrame:
        return pd.DataFrame()

    def build_sections(self, cmp_groups: list[GroupedTestRuns]) -> list[ComparisonSection]:
        return []


class RenderableComparisonReport(MyComparisonReport):
    def load_test_runs(self) -> None:
        """Keep test-provided runs instead of loading result directories."""

    def build_sections(self, cmp_groups: list[GroupedTestRuns]) -> list[ComparisonSection]:
        return [
            ComparisonSection(
                group=group,
                dfs=[pd.DataFrame({"size": [1, 2, 4], "value": [10, 20, 40]}) for _ in group.items],
                title="Throughput",
                info_columns=["size"],
                data_columns=["value"],
                y_axis_label="Requests/s",
            )
            for group in cmp_groups
        ]


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


def test_v2_jinja_template_path(cmp_report: MyComparisonReport) -> None:
    full_path = cmp_report.template_path / cmp_report.v2_template_name
    assert full_path.exists()
    assert full_path.is_file()


def test_generate_writes_legacy_and_v2_reports(slurm_system: SlurmSystem, nccl_tr: TestRun) -> None:
    slurm_system.output_path.mkdir(parents=True)
    report = RenderableComparisonReport(
        slurm_system,
        TestScenario(name="comparison", test_runs=[]),
        slurm_system.output_path,
        ComparisonReportConfig(enable=True),
    )
    report.trs = [nccl_tr, copy.deepcopy(nccl_tr)]

    report.generate()

    legacy_path = slurm_system.output_path / "comparison_report.html"
    v2_path = slurm_system.output_path / "comparison_report_v2.html"
    assert legacy_path.exists()
    assert v2_path.exists()

    v2_content = v2_path.read_text()
    assert "Show full labels" not in v2_content
    assert "js-label-toggle" not in v2_content
    assert "chart.js@4.4.3" in v2_content
    assert "chartjs-plugin-zoom@2.2.0" in v2_content
    assert "overlap exactly" not in v2_content
    assert "Reset view" in v2_content
    assert "fallback.hidden = true" in v2_content
    assert ".chart-shell.is-enhanced ~ .chart-fallback" in v2_content
    assert "dataset.borderDash = []" in v2_content
    assert "overflow-x: auto" in v2_content
    assert "width: fit-content" in v2_content
    assert "min-width: 100%" not in v2_content
    assert "const isShiftWheel = event.shiftKey" in v2_content
    assert "event.deltaY) >= Math.abs(event.deltaX)" in v2_content
    assert "wheel: {\n                                    enabled: false" in v2_content
    assert "Shift + wheel or pinch to zoom" in v2_content
    assert "max-width: 18rem" in v2_content
    assert "width: clamp(16rem, 24vw, 32rem)" not in v2_content
    assert 'mode: "xy"' in v2_content
    assert "indexedCategoryLimits" not in v2_content
    assert "useAutoLogScale" in v2_content
    assert "js-column-picker" in v2_content
    assert "js-column-toggle" in v2_content
    assert "Show all" in v2_content
    assert "setColumnVisibility" in v2_content
    assert "cell.hidden = !visible" in v2_content
    assert "Columns (" in v2_content
    assert nccl_tr.name in v2_content


def test_v2_payload_uses_compact_labels_and_structured_differences(
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

    chart = cmp_report._build_v2_chart(section, 0)
    table = cmp_report._build_v2_table(section)

    assert chart["datasets"][0]["label"] == "case-a"
    assert "fullLabel" not in chart["datasets"][0]
    assert "borderColor" not in chart["datasets"][0]
    assert table["data_headers"][0]["name"] == "case-a"
    assert table["data_headers"][0]["differences_yaml"] == (
        f'docker_image_url: "{long_image}"\nprefill:\n  gpu_ids: ["0", "1"]\n  tensor_parallel_size: 2'
    )


def test_v2_indexed_category_axis_uses_display_labels(cmp_report: MyComparisonReport, nccl_tr: TestRun) -> None:
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

    chart = cmp_report._build_v2_chart(section, 0)

    assert chart["x_axis_type"] == "indexed_category"
    assert chart["x_axis_label"] == "Message size"
    assert chart["labels"] == ["256B", "1KB", "4KB"]
    assert chart["datasets"][0]["data"] == [1.0, 2.0, 3.0]


def test_v2_auto_y_axis_is_in_payload(cmp_report: MyComparisonReport, nccl_tr: TestRun) -> None:
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

    chart = cmp_report._build_v2_chart(section, 0)

    assert chart["y_axis_type"] == "auto"


@pytest.mark.parametrize(
    ("report_cls", "legacy_name", "v2_name"),
    [
        (NIXLBenchComparisonReport, "nixl_comparison.html", "nixl_comparison_v2.html"),
        (NixlEPComparisonReport, "nixl_ep_comparison.html", "nixl_ep_comparison_v2.html"),
        (NcclComparisonReport, "nccl_comparison.html", "nccl_comparison_v2.html"),
        (OSUBenchComparisonReport, "osu_bench_comparison.html", "osu_bench_comparison_v2.html"),
        (VLLMComparisonReport, "vllm_comparison.html", "vllm_comparison_v2.html"),
        (SGLangComparisonReport, "sglang_comparison.html", "sglang_comparison_v2.html"),
        (AIDynamoComparisonReport, "ai_dynamo_comparison.html", "ai_dynamo_comparison_v2.html"),
    ],
)
def test_v2_report_file_names(
    slurm_system: SlurmSystem,
    report_cls: type[ComparisonReport],
    legacy_name: str,
    v2_name: str,
) -> None:
    report = report_cls(
        slurm_system,
        TestScenario(name="comparison", test_runs=[]),
        slurm_system.output_path,
        ComparisonReportConfig(enable=True),
    )

    assert report.report_file_name == legacy_name
    assert report.v2_report_file_name == v2_name


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
