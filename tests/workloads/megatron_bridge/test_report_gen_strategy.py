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

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from cloudai import TestRun
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.megatron_bridge import GOLDEN_VALUES_FILENAME, MegatronBridgeReportGenerationStrategy


@pytest.fixture
def mb_tr(tmp_path: Path) -> TestRun:
    tr = TestRun(name="megatron_bridge", test=Mock(), num_nodes=1, nodes=[], output_path=tmp_path)
    metrics = {
        "some_other_data": 1.23,
        "1": {"elapsed time per iteration (ms)": 1000, "GPU utilization": 1.23},
        "0": {"elapsed time per iteration (ms)": 1000, "GPU utilization": 1.23},
    }
    metrics_folder = tr.output_path / "experiments" / "some_experiment"
    metrics_folder.mkdir(parents=True)
    (metrics_folder / GOLDEN_VALUES_FILENAME).write_text(json.dumps(metrics))
    return tr


def test_megatron_bridge_can_handle_directory(slurm_system: SlurmSystem, mb_tr: TestRun) -> None:
    strategy = MegatronBridgeReportGenerationStrategy(slurm_system, mb_tr)
    assert strategy.can_handle_directory()


def test_megatron_bridge_extract_and_generate_report(slurm_system: SlurmSystem, mb_tr: TestRun) -> None:
    strategy = MegatronBridgeReportGenerationStrategy(slurm_system, mb_tr)
    strategy.generate_report()
    report_path = mb_tr.output_path / "report.txt"
    assert report_path.is_file()
    content = report_path.read_text()
    assert (
        content
        == f"""
Source log: {mb_tr.output_path}/experiments/some_experiment/{GOLDEN_VALUES_FILENAME}

Step Time (s)
  avg: 1.0
  median: 1.0
  min: 1.0
  max: 1.0
  std: 0.0

TFLOP/s per GPU
  avg: 1.23
  median: 1.23
  min: 1.23
  max: 1.23
  std: 0.0
""".lstrip()
    )
