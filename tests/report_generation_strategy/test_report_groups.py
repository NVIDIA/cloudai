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

import bokeh.plotting as bk
import pandas as pd
from rich.table import Table

from cloudai import TestRun
from cloudai.core import TestScenario
from cloudai.report_generator.comparison_report import ComparisonReport, ComparisonReportConfig
from cloudai.report_generator.groups import GroupedTestRuns
from cloudai.report_generator.util import diff_comparison_values
from cloudai.systems.slurm import SlurmSystem


class GroupingComparisonReport(ComparisonReport):
    def extract_data_as_df(self, tr: TestRun) -> pd.DataFrame:
        return pd.DataFrame()

    def create_tables(self, cmp_groups: list[GroupedTestRuns]) -> list[Table]:
        return []

    def create_charts(self, cmp_groups: list[GroupedTestRuns]) -> list[bk.figure]:
        return []


def _comparison_report(
    slurm_system: SlurmSystem, trs: list[TestRun], group_by: list[str]
) -> GroupingComparisonReport:
    report = GroupingComparisonReport(
        slurm_system,
        TestScenario(name="comparison", test_runs=[]),
        slurm_system.output_path,
        ComparisonReportConfig(enable=True, group_by=group_by),
    )
    report.trs = trs
    return report


class TestGrouping:
    def test_single_tr(self, slurm_system: SlurmSystem, nccl_tr: TestRun) -> None:
        groups = _comparison_report(slurm_system, [nccl_tr], group_by=[]).group_test_runs()
        assert len(groups) == 1
        assert groups[0].name == "all-in-one"
        assert groups[0].items[0].name == "0.0"

    def test_multiple_trs_no_group_by_fields_same_trs(
        self, slurm_system: SlurmSystem, nccl_tr: TestRun
    ) -> None:
        groups = _comparison_report(slurm_system, [nccl_tr, nccl_tr], group_by=[]).group_test_runs()
        assert len(groups) == 1
        assert groups[0].name == "all-in-one"
        assert groups[0].items[0].name == "0.0"
        assert groups[0].items[1].name == "0.1"

    def test_multiple_trs_no_group_by_fields(self, slurm_system: SlurmSystem, nccl_tr: TestRun) -> None:
        nccl1 = copy.deepcopy(nccl_tr)
        nccl2 = copy.deepcopy(nccl_tr)
        nccl1.test.cmd_args.subtest_name = "all_gather_perf"
        nccl2.test.cmd_args.subtest_name = "all_reduce_perf"
        groups = _comparison_report(slurm_system, [nccl1, nccl2], group_by=[]).group_test_runs()
        assert len(groups) == 1
        assert groups[0].name == "all-in-one"
        assert groups[0].items[0].name == "subtest_name=all_gather_perf"
        assert groups[0].items[1].name == "subtest_name=all_reduce_perf"

    def test_group_by_one_field(self, slurm_system: SlurmSystem, nccl_tr: TestRun) -> None:
        nccl1 = copy.deepcopy(nccl_tr)
        nccl2 = copy.deepcopy(nccl_tr)
        nccl1.test.cmd_args.subtest_name = "all_gather_perf"
        nccl2.test.cmd_args.subtest_name = "all_reduce_perf"

        groups = _comparison_report(slurm_system, [nccl1, nccl2], group_by=["subtest_name"]).group_test_runs()

        assert len(groups) == 2
        assert groups[0].name == "subtest_name=all_gather_perf"
        assert groups[1].name == "subtest_name=all_reduce_perf"
        assert groups[0].items[0].name == "0.0"
        assert groups[1].items[0].name == "1.0"

    def test_group_by_two_fields(self, slurm_system: SlurmSystem, nccl_tr: TestRun) -> None:
        nccl_tr.test.cmd_args.subtest_name = ["all_gather_perf", "all_reduce_perf"]
        nccl_tr.test.extra_env_vars["NCCL_IB_SPLIT_DATA_ON_QPS"] = ["0", "1"]
        trs: list[TestRun] = [nccl_tr.apply_params_set(combination) for combination in nccl_tr.all_combinations]

        groups = _comparison_report(
            slurm_system,
            trs,
            ["subtest_name", "extra_env_vars.NCCL_IB_SPLIT_DATA_ON_QPS"],
        ).group_test_runs()

        assert len(groups) == 4
        assert all(len(group.items) == 1 for group in groups)
        assert groups[0].name == "subtest_name=all_gather_perf NCCL_IB_SPLIT_DATA_ON_QPS=0"
        assert groups[1].name == "subtest_name=all_gather_perf NCCL_IB_SPLIT_DATA_ON_QPS=1"
        assert groups[2].name == "subtest_name=all_reduce_perf NCCL_IB_SPLIT_DATA_ON_QPS=0"
        assert groups[3].name == "subtest_name=all_reduce_perf NCCL_IB_SPLIT_DATA_ON_QPS=1"

    def test_multiple_trs_in_a_group(self, slurm_system: SlurmSystem, nccl_tr: TestRun) -> None:
        nccl_tr.test.cmd_args.subtest_name = ["all_gather_perf", "all_reduce_perf"]
        nccl_tr.test.extra_env_vars["NCCL_IB_SPLIT_DATA_ON_QPS"] = ["0", "1"]
        trs: list[TestRun] = [nccl_tr.apply_params_set(combination) for combination in nccl_tr.all_combinations]

        groups = _comparison_report(slurm_system, trs, group_by=["subtest_name"]).group_test_runs()

        assert len(groups) == 2

        assert groups[0].name == "subtest_name=all_gather_perf"
        assert len(groups[0].items) == 2
        assert groups[0].items[0].name == "NCCL_IB_SPLIT_DATA_ON_QPS=0"
        assert groups[0].items[1].name == "NCCL_IB_SPLIT_DATA_ON_QPS=1"

        assert groups[1].name == "subtest_name=all_reduce_perf"
        assert len(groups[1].items) == 2
        assert groups[1].items[0].name == "NCCL_IB_SPLIT_DATA_ON_QPS=0"
        assert groups[1].items[1].name == "NCCL_IB_SPLIT_DATA_ON_QPS=1"


class TestComparisonValues:
    def test_diff_comparison_values_normalizes_numeric_equivalents(self) -> None:
        diff = diff_comparison_values([{"max_concurrency": 1}, {"max_concurrency": 1.0}])

        assert diff == {}

    def test_diff_comparison_values_normalizes_mapping_order(self) -> None:
        diff = diff_comparison_values(
            [
                {"extra": {"backend": "nixl", "mode": "decode"}},
                {"extra": {"mode": "decode", "backend": "nixl"}},
            ]
        )

        assert diff == {}

    def test_diff_comparison_values_normalizes_nested_numeric_equivalents(self) -> None:
        diff = diff_comparison_values(
            [
                {"extra": {"limits": [1, {"max": 2.0}], "enabled": True}},
                {"extra": {"enabled": True, "limits": [1.0, {"max": 2}]}},
            ]
        )

        assert diff == {}

    def test_default_comparison_values_include_cmd_args(
        self, slurm_system: SlurmSystem, nccl_tr: TestRun
    ) -> None:
        nccl1 = copy.deepcopy(nccl_tr)
        nccl2 = copy.deepcopy(nccl_tr)
        nccl1.test.cmd_args.subtest_name = "all_gather_perf"
        nccl2.test.cmd_args.subtest_name = "all_reduce_perf"

        report = _comparison_report(slurm_system, [nccl1, nccl2], group_by=[])
        diff = diff_comparison_values([report.comparison_values(nccl1), report.comparison_values(nccl2)])

        assert diff == {"subtest_name": ["all_gather_perf", "all_reduce_perf"]}

    def test_default_comparison_values_include_num_nodes(
        self, slurm_system: SlurmSystem, nccl_tr: TestRun
    ) -> None:
        nccl1 = copy.deepcopy(nccl_tr)
        nccl2 = copy.deepcopy(nccl_tr)
        nccl1.num_nodes = 1
        nccl2.num_nodes = 2

        report = _comparison_report(slurm_system, [nccl1, nccl2], group_by=[])
        diff = diff_comparison_values([report.comparison_values(nccl1), report.comparison_values(nccl2)])
        assert diff == {"NUM_NODES": [1, 2]}

    def test_default_comparison_values_include_extra_env_vars(
        self, slurm_system: SlurmSystem, nccl_tr: TestRun
    ) -> None:
        nccl1 = copy.deepcopy(nccl_tr)
        nccl2 = copy.deepcopy(nccl_tr)
        nccl1.test.extra_env_vars["NCCL_IB_SPLIT_DATA_ON_QPS"] = "0"
        nccl2.test.extra_env_vars["NCCL_IB_SPLIT_DATA_ON_QPS"] = "1"

        report = _comparison_report(slurm_system, [nccl1, nccl2], group_by=[])
        diff = diff_comparison_values([report.comparison_values(nccl1), report.comparison_values(nccl2)])
        assert diff == {"extra_env_vars.NCCL_IB_SPLIT_DATA_ON_QPS": ["0", "1"]}
