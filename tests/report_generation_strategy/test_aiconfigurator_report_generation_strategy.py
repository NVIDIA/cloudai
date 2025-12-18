# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cloudai.core import METRIC_ERROR, TestRun
from cloudai.systems.standalone import StandaloneSystem
from cloudai.workloads.aiconfig import (
    AiconfiguratorCmdArgs,
    AiconfiguratorReportGenerationStrategy,
    AiconfiguratorTestDefinition,
)
from cloudai.workloads.aiconfig.aiconfigurator import Agg


def _make_tr(tmp_path: Path) -> TestRun:
    tdef = AiconfiguratorTestDefinition(
        name="aiconfig",
        description="desc",
        test_template_name="Aiconfigurator",
        cmd_args=AiconfiguratorCmdArgs(
            model_name="m",
            system="s",
            isl=1,
            osl=2,
            agg=Agg(batch_size=1, ctx_tokens=16),
        ),
    )
    return TestRun(name="tr", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "out")


def test_can_handle_directory_when_report_exists(tmp_path: Path, standalone_system: StandaloneSystem) -> None:
    tr = _make_tr(tmp_path)
    tr.output_path.mkdir(parents=True, exist_ok=True)
    (tr.output_path / "report.json").write_text("{}", encoding="utf-8")

    strategy = AiconfiguratorReportGenerationStrategy(standalone_system, tr)
    assert strategy.can_handle_directory() is True


def test_generate_report_writes_summary(tmp_path: Path, standalone_system: StandaloneSystem) -> None:
    tr = _make_tr(tmp_path)
    tr.output_path.mkdir(parents=True, exist_ok=True)
    payload = {"ttft_ms": 10.0, "tpot_ms": 2.0, "tokens_per_s_per_gpu": 3.0, "tokens_per_s_per_user": 4.0, "oom": False}
    (tr.output_path / "report.json").write_text(json.dumps(payload), encoding="utf-8")

    strategy = AiconfiguratorReportGenerationStrategy(standalone_system, tr)
    strategy.generate_report()

    summary = (tr.output_path / "summary.txt").read_text(encoding="utf-8")
    assert "ttft_ms: 10.0" in summary
    assert "tpot_ms: 2.0" in summary
    assert "tokens_per_s_per_gpu: 3.0" in summary
    assert "tokens_per_s_per_user: 4.0" in summary
    assert "oom: False" in summary


def test_get_metric_default_prefers_throughput(tmp_path: Path, standalone_system: StandaloneSystem) -> None:
    tr = _make_tr(tmp_path)
    tr.output_path.mkdir(parents=True, exist_ok=True)
    (tr.output_path / "report.json").write_text(json.dumps({"tokens_per_s_per_gpu": 123.0}), encoding="utf-8")

    strategy = AiconfiguratorReportGenerationStrategy(standalone_system, tr)
    assert strategy.get_metric("default") == 123.0


def test_get_metric_default_falls_back_to_inverse_latency(tmp_path: Path, standalone_system: StandaloneSystem) -> None:
    tr = _make_tr(tmp_path)
    tr.output_path.mkdir(parents=True, exist_ok=True)
    (tr.output_path / "report.json").write_text(json.dumps({"tpot_ms": 2.0}), encoding="utf-8")

    strategy = AiconfiguratorReportGenerationStrategy(standalone_system, tr)
    assert pytest.approx(strategy.get_metric("default"), rel=1e-6) == 0.5


def test_load_results_falls_back_to_stdout_last_json(tmp_path: Path, standalone_system: StandaloneSystem) -> None:
    tr = _make_tr(tmp_path)
    tr.output_path.mkdir(parents=True, exist_ok=True)
    (tr.output_path / "stdout.txt").write_text('noise\n{"tokens_per_s_per_user": 7}\n', encoding="utf-8")

    strategy = AiconfiguratorReportGenerationStrategy(standalone_system, tr)
    assert strategy.get_metric("tokens_per_s_per_user") == 7.0


def test_get_metric_unknown_returns_error(tmp_path: Path, standalone_system: StandaloneSystem) -> None:
    tr = _make_tr(tmp_path)
    tr.output_path.mkdir(parents=True, exist_ok=True)
    (tr.output_path / "report.json").write_text(json.dumps({"ttft_ms": 1.0}), encoding="utf-8")

    strategy = AiconfiguratorReportGenerationStrategy(standalone_system, tr)
    assert strategy.get_metric("nonexistent") == METRIC_ERROR
