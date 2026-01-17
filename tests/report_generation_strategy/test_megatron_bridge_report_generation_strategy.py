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
from unittest.mock import Mock

import pytest

from cloudai import TestRun
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.megatron_bridge import MegatronBridgeReportGenerationStrategy


@pytest.fixture
def mb_tr(tmp_path: Path) -> TestRun:
    tr = TestRun(name="megatron_bridge", test=Mock(), num_nodes=1, nodes=[], output_path=tmp_path)
    log_content = "\n".join(
        [
            "ain_fp8_mx/0 Step Time : 9.09s GPU utilization: 663.5MODEL_TFLOP/s/GPU",
            "",
            "ain_fp8_mx/0  [2025-12-22 15:18:33] iteration       50/      50 | consumed samples:        25600 | "
            "elapsed time per iteration (ms): 9089.0 | learning rate: 3.000000E-05 | global batch size:   512 | "
            "lm loss: 8.114214E+00 | load_balancing_loss: 1.000000E+00 | loss scale: 1.0 | grad norm: 0.042 | "
            "number of skipped iterations:   0 | number of nan iterations:   0 |",
            "",
        ]
    )
    (tr.output_path / "cloudai_megatron_bridge_launcher.log").write_text(log_content)
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
    assert "Step Time" in content
    assert "TFLOP/s per GPU" in content
