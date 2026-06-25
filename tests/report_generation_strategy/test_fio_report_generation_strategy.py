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

import pytest

from cloudai.core import METRIC_ERROR, TestRun
from cloudai.systems.standalone import StandaloneSystem
from cloudai.workloads.fio import FioCmdArgs, FioReportGenerationStrategy, FioTestDefinition

FIO_STDOUT = """
job1: (groupid=0, jobs=1): err= 0: pid=1
  write: IOPS=100, BW=10MiB/s (10MB/s)(80.0MiB/14msec)
     lat (usec): min=1, max=3, avg=2.00, stdev=0.1
job2: (groupid=0, jobs=1): err= 0: pid=2
  write: IOPS=200, BW=20MiB/s (20MB/s)(80.0MiB/14msec)
     lat (usec): min=2, max=4, avg=3.00, stdev=0.1
job3: (groupid=0, jobs=1): err= 0: pid=3
  read: IOPS=300, BW=30MiB/s (30MB/s)(80.0MiB/14msec)
     lat (usec): min=3, max=5, avg=4.00, stdev=0.1
"""


def test_fio_report_happy_path(standalone_system: StandaloneSystem, tmp_path: Path) -> None:
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
    assert strategy.get_metric("default") == 60
    assert strategy.get_metric("write_iops") == 200
    assert strategy.get_metric("read_bw") == 30
    assert strategy.get_metric("unknown") is METRIC_ERROR

    strategy.generate_report()

    csv_path = tmp_path / "fio_summary.csv"
    assert csv_path.is_file()
    csv_text = csv_path.read_text()
    assert "operation,iops,bw,bw_unit,latency_avg,latency_unit" in csv_text
    assert "read,300.0,30.0,MiB/s,4.0,usec" in csv_text


def _strategy_for_metric(
    standalone_system: StandaloneSystem,
    tmp_path: Path,
    metric_operation: str = "all",
    metric_name: str = "bw",
    metric_aggregate: str = "sum",
) -> FioReportGenerationStrategy:
    tdef = FioTestDefinition(
        name="fio",
        description="fio test",
        test_template_name="Fio",
        cmd_args=FioCmdArgs(
            args={"name": "smoke"},
            metric_operation=metric_operation,
            metric_name=metric_name,
            metric_aggregate=metric_aggregate,
        ),
    )
    tr = TestRun(name="fio", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)
    (tmp_path / "stdout.txt").write_text(FIO_STDOUT)
    return FioReportGenerationStrategy(standalone_system, tr)


@pytest.mark.parametrize(
    ("metric_operation", "metric_name", "metric_aggregate", "expected"),
    [
        ("write", "bw", "sum", 30),
        ("write", "iops", "mean", 150),
        ("all", "bw", "min", 10),
        ("all", "bw", "max", 30),
        ("all", "bw", "first", 10),
        ("all", "latency", "mean", 3),
    ],
)
def test_default_metric_configuration(
    standalone_system: StandaloneSystem,
    tmp_path: Path,
    metric_operation: str,
    metric_name: str,
    metric_aggregate: str,
    expected: float,
) -> None:
    strategy = _strategy_for_metric(
        standalone_system,
        tmp_path,
        metric_operation=metric_operation,
        metric_name=metric_name,
        metric_aggregate=metric_aggregate,
    )

    assert strategy.get_metric("default") == expected
