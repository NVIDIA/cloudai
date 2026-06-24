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

from cloudai.core import METRIC_ERROR, TestRun
from cloudai.systems.standalone import StandaloneSystem
from cloudai.workloads.fio import FioCmdArgs, FioReportGenerationStrategy, FioTestDefinition
from cloudai.workloads.fio.report_generation_strategy import extract_fio_data

FIO_STDOUT = """
job1: (groupid=0, jobs=1): err= 0: pid=15642: Thu Jun 11 21:06:45 2026
  read: IOPS=315, BW=2524MiB/s (2647MB/s)(24.7GiB/10012msec)
    clat (usec): min=6941, max=30695, avg=14504.83, stdev=2564.83
     lat (usec): min=6963, max=30717, avg=14527.39, stdev=2564.96
  write: IOPS=1.2k, BW=2633MiB/s (2761MB/s)(25.7GiB/10012msec)
    clat (usec): min=1735, max=26776, avg=9949.52, stdev=2895.16
     lat (usec): min=2190, max=27386, avg=10374.89, stdev=2885.57
"""


def test_extract_fio_data_parses_iops_bw_and_latency(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.txt"
    stdout.write_text(FIO_STDOUT)

    rows = extract_fio_data(stdout)

    assert len(rows) == 2
    assert rows[0].operation == "read"
    assert rows[0].iops == 315
    assert rows[0].bw == 2524
    assert rows[0].latency_avg == 14527.39
    assert rows[1].operation == "write"
    assert rows[1].iops == 1200


def test_fio_report_writes_summary_csv(standalone_system: StandaloneSystem, tmp_path: Path) -> None:
    tdef = FioTestDefinition(
        name="fio",
        description="fio test",
        test_template_name="Fio",
        cmd_args=FioCmdArgs(args={"name": "smoke"}),
    )
    tr = TestRun(name="fio", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)
    (tmp_path / "stdout.txt").write_text(FIO_STDOUT)
    strategy = FioReportGenerationStrategy(standalone_system, tr)

    assert strategy.can_handle_directory()
    assert strategy.get_metric("read_bw") == 2524
    assert strategy.get_metric("write_iops") == 1200
    assert strategy.get_metric("unknown") is METRIC_ERROR

    strategy.generate_report()

    csv_path = tmp_path / "fio_summary.csv"
    assert csv_path.is_file()
    assert "operation,iops,bw,bw_unit,latency_avg,latency_unit" in csv_path.read_text()
