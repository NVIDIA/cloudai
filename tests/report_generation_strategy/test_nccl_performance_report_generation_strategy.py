# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pandas as pd
import pytest

from cloudai import TestRun
from cloudai.core import METRIC_ERROR
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.util.lazy_imports import lazy
from cloudai.workloads.nccl_test import NcclTestPerformanceReportGenerationStrategy
from cloudai.workloads.nccl_test.performance_report_generation_strategy import _parse_device_info


@pytest.fixture
def report_strategy(slurm_system: SlurmSystem, nccl_tr: TestRun) -> NcclTestPerformanceReportGenerationStrategy:
    return NcclTestPerformanceReportGenerationStrategy(slurm_system, nccl_tr)


def test_can_handle_directory(report_strategy: NcclTestPerformanceReportGenerationStrategy) -> None:
    assert report_strategy.can_handle_directory() is True


def test_generate_performance_report(
    report_strategy: NcclTestPerformanceReportGenerationStrategy, nccl_tr: TestRun
) -> None:
    report_strategy.generate_report()

    csv_report_path = nccl_tr.output_path / "cloudai_nccl_test_csv_report.csv"
    assert csv_report_path.is_file(), "CSV report was not generated."

    df = pd.read_csv(csv_report_path)
    assert not df.empty, "CSV report is empty."

    # Validate data types
    assert df["Size (B)"].dtype == int, "Size (B) is not an integer."
    assert df["Time (us) Out-of-place"].dtype == float, "Time (us) Out-of-place is not a float."
    assert df["Time (us) In-place"].dtype == float, "Time (us) In-place is not a float."

    # Validate human-readable sizes
    assert df.iloc[0]["Size Human-readable"] == "976.6KB"
    assert df.iloc[-1]["Size Human-readable"] == "11.4MB"

    # Validate first entry
    assert df.iloc[0]["Size (B)"] == 1000000
    assert df.iloc[0]["Algbw (GB/s) Out-of-place"] == 10.10
    assert df.iloc[0]["Busbw (GB/s) Out-of-place"] == 20.20

    # Validate last entry
    assert df.iloc[-1]["Size (B)"] == 12000000
    assert df.iloc[-1]["Algbw (GB/s) Out-of-place"] == 120.30
    assert df.iloc[-1]["Busbw (GB/s) Out-of-place"] == 130.40

    # Ensure extracted values match expectations
    assert df["GPU Type"].iloc[0] == "H100", "GPU Type was not extracted correctly."
    assert df["Devices per Node"].iloc[0] == 8, "Devices per Node is incorrect."
    assert df["Ranks"].iloc[0] == 16, "Ranks is incorrect."


def test_parse_gpu_types(report_strategy: NcclTestPerformanceReportGenerationStrategy, tmp_path: Path) -> None:
    test_cases = [
        ("NVIDIA Tesla V100-PCIE-32GB", "Tesla V100-PCIE-32GB"),
        ("NVIDIA H100-SXM5-80GB", "H100-SXM5-80GB"),
        ("NVIDIA A100-SXM4-40GB", "A100-SXM4-40GB"),
        ("NVIDIA Tesla T4", "Tesla T4"),
        ("NVIDIA H100", "H100"),
    ]

    for gpu_line, expected_type in test_cases:
        stdout_content = f"""# Rank  0 Group  0 Pid 1000 on node1 device  0 [0xaa] {gpu_line}
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
         128            32     float     sum      -1    21.89    0.01    0.01      0    17.61    0.01    0.01      0
# Avg bus bandwidth    : 111.111
"""
        test_file = tmp_path / "test_stdout.txt"
        test_file.write_text(stdout_content)

        with test_file.open("r", encoding="utf-8") as file:
            _, gpu_type, _ = _parse_device_info(file)
            assert gpu_type == expected_type, f"Failed to parse GPU type for {gpu_line}"


@pytest.mark.parametrize(
    "metric,ref_values",
    [
        ("default", [1.12, 2.23, 13.14]),
        ("latency-in-place", [1.12, 2.23, 13.14]),
        ("latency-out-of-place", [1.11, 2.22, 13.13]),
    ],
)
def test_get_metric(
    report_strategy: NcclTestPerformanceReportGenerationStrategy, metric: str, ref_values: list[float]
) -> None:
    res = report_strategy.get_metric(metric)
    assert res is not None and res != METRIC_ERROR
    assert res == lazy.np.mean(ref_values)
