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
import logging
from typing import ClassVar, Optional

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy

from .aiconfigurator import AiconfiguratorTestDefinition


class AiconfiguratorReportGenerationStrategy(ReportGenerationStrategy):
    """Generate metrics from Aiconfigurator predictor outputs."""

    metrics: ClassVar[list[str]] = [
        "default",
        "ttft_ms",
        "tpot_ms",
        "tokens_per_s_per_gpu",
        "tokens_per_s_per_user",
    ]

    def can_handle_directory(self) -> bool:
        return isinstance(self.test_run.test, AiconfiguratorTestDefinition) and (
            (self.test_run.output_path / "report.json").is_file()
            or (self.test_run.output_path / "stdout.txt").is_file()
        )

    def _load_results(self) -> Optional[dict]:
        result_path = self.test_run.output_path / "report.json"
        if result_path.is_file():
            try:
                with result_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logging.debug(f"Failed to parse JSON from {result_path}: {e}")

        stdout_path = self.test_run.output_path / "stdout.txt"
        if stdout_path.is_file():
            try:
                with stdout_path.open("r", encoding="utf-8", errors="ignore") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                for line in reversed(lines):
                    if line.startswith("{") and line.endswith("}"):
                        return json.loads(line)
            except Exception:
                pass
        return None

    def generate_report(self) -> None:
        data = self._load_results()
        if not data:
            logging.error(f"No Aiconfigurator results found under {self.test_run.output_path}. Skipping report.")
            return

        summary_path = self.test_run.output_path / "summary.txt"
        try:
            with summary_path.open("w", encoding="utf-8") as f:
                for key in [
                    "ttft_ms",
                    "tpot_ms",
                    "tokens_per_s_per_gpu",
                    "tokens_per_s_per_user",
                    "oom",
                ]:
                    if key in data:
                        f.write(f"{key}: {data[key]}\n")
            logging.info(f"Aiconfigurator summary written to {summary_path}")
        except Exception as e:
            logging.error(f"Failed to write summary: {e}")

    def get_metric(self, metric: str) -> float:
        data = self._load_results()
        if not data:
            return METRIC_ERROR

        if metric == "default":
            for k in ("tokens_per_s_per_gpu", "tokens_per_s_per_user"):
                v = data.get(k)
                if isinstance(v, (int, float)):
                    return float(v)

            for k in ("tpot_ms", "ttft_ms"):
                v = data.get(k)
                if isinstance(v, (int, float)):
                    return float(1.0 / max(float(v), 1e-9))
            return METRIC_ERROR

        if metric in {"ttft_ms", "tpot_ms", "tokens_per_s_per_gpu", "tokens_per_s_per_user"}:
            v = data.get(metric)
            return float(v) if isinstance(v, (int, float)) else METRIC_ERROR

        return METRIC_ERROR
