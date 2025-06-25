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
from unittest.mock import Mock

import pandas as pd
import pytest

from cloudai import Test, TestRun
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition, NcclTestPerformanceReportGenerationStrategy


@pytest.fixture
def nccl_tr(tmp_path: Path) -> TestRun:
    test = Test(
        test_definition=NCCLTestDefinition(
            name="nccl",
            description="desc",
            test_template_name="t",
            cmd_args=NCCLCmdArgs(),
        ),
        test_template=Mock(),
    )
    tr = TestRun(name="nccl_test", test=test, num_nodes=2, nodes=[], output_path=tmp_path)

    stdout_content = """# Rank  0 Group  0 Pid 1000 on node1 device  0 [0xaa] NVIDIA H100
# Rank  1 Group  0 Pid 1001 on node1 device  1 [0xbb] NVIDIA H100
# Rank  2 Group  0 Pid 1002 on node1 device  2 [0xcc] NVIDIA H100
# Rank  3 Group  0 Pid 1003 on node1 device  3 [0xdd] NVIDIA H100
# Rank  4 Group  0 Pid 1004 on node1 device  4 [0xee] NVIDIA H100
# Rank  5 Group  0 Pid 1005 on node1 device  5 [0xff] NVIDIA H100
# Rank  6 Group  0 Pid 1006 on node1 device  6 [0x11] NVIDIA H100
# Rank  7 Group  0 Pid 1007 on node1 device  7 [0x22] NVIDIA H100
# Rank  8 Group  0 Pid 2000 on node2 device  0 [0xaa] NVIDIA H100
# Rank  9 Group  0 Pid 2001 on node2 device  1 [0xbb] NVIDIA H100
# Rank 10 Group  0 Pid 2002 on node2 device  2 [0xcc] NVIDIA H100
# Rank 11 Group  0 Pid 2003 on node2 device  3 [0xdd] NVIDIA H100
# Rank 12 Group  0 Pid 2004 on node2 device  4 [0xee] NVIDIA H100
# Rank 13 Group  0 Pid 2005 on node2 device  5 [0xff] NVIDIA H100
# Rank 14 Group  0 Pid 2006 on node2 device  6 [0x11] NVIDIA H100
# Rank 15 Group  0 Pid 2007 on node2 device  7 [0x22] NVIDIA H100
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
     1000000       1000000     float     sum      -1    1.11   10.10   20.20      0    1.12   10.11   20.21      0
     2000000       2000000     float     sum      -1    2.22   20.20   30.30      0    2.23   20.21   30.31      0
     12000000      12000000     float     sum      -1   13.13  120.30  130.40      0   13.14  120.31  130.41      0
# Avg bus bandwidth    : 111.111
"""
    (tr.output_path / "stdout.txt").write_text(stdout_content)

    return tr


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
            _, gpu_type, _ = report_strategy._parse_device_info(file)
            assert gpu_type == expected_type, f"Failed to parse GPU type for {gpu_line}"
