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

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from cloudai.core import TestRun, TestScenario
from cloudai.report_generator.comparison_report import ComparisonReportConfig
from cloudai.report_generator.groups import GroupedTestRuns, TRGroupItem
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.osu_bench.osu_comparison_report import OSUBenchComparisonReport


@pytest.mark.parametrize("metric", ["avg_lat", "mb_sec", "messages_sec"])
@pytest.mark.parametrize("sizes", [[0, 1, 2, 4, 8], [2, 4, 8, 1024, 4194304]])
def test_v2_axis_labels_match_measured_sizes(
    metric: str, sizes: list[int], tmp_path: Path, slurm_system: SlurmSystem
) -> None:
    values = [float(index + 1) for index in range(len(sizes))]
    pd.DataFrame({"size": sizes, metric: values}).to_csv(tmp_path / "osu_bench.csv", index=False)
    tr = TestRun(name="osu", test=Mock(), num_nodes=2, nodes=[], output_path=tmp_path)
    report = OSUBenchComparisonReport(
        slurm_system,
        TestScenario(name="osu", test_runs=[]),
        tmp_path,
        ComparisonReportConfig(enable=True, group_by=[]),
    )
    group = GroupedTestRuns(name="all-in-one", items=[TRGroupItem(name="case-a", tr=tr)])

    sections = report.build_sections([group])
    assert len(sections) == 1
    chart = report._build_sections_v2(sections)[0]["chart"]

    assert chart["x_axis_type"] == "indexed_category"
    assert chart["labels"] == [str(size) for size in sizes]
    assert chart["datasets"][0]["data"] == values
