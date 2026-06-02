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

from __future__ import annotations

from cloudai.workloads.ai_dynamo.report_generation_strategy import AIDynamoReportGenerationStrategy

from .dynamo_mocker import DynamoMockerTestDefinition


class DynamoMockerReportGenerationStrategy(AIDynamoReportGenerationStrategy):
    """
    Generate metrics from Dynamo Mocker benchmark output (benchmark_report.csv).

    Inherits genai_perf CSV parsing from AIDynamoReportGenerationStrategy.
    The CSV is written by genai_perf.sh with columns: Metric, avg, min, max, p99, ...

    Metric names to use in agent_metrics / get_metric():
      - "Time To First Token (ms)"           — TTFT average
      - "Inter Token Latency (ms)"           — TPOT / ITL average
      - "Output Token Throughput (tokens/sec)"
      - "Request Throughput (per sec)"
      - "Output tokens per second per gpu"   — added by genai_perf.sh
      Use "genai_perf:<metric name>:<column>" for non-avg columns, e.g.
      "genai_perf:Time To First Token (ms):p99"
    """

    def can_handle_directory(self) -> bool:
        return (
            isinstance(self.test_run.test, DynamoMockerTestDefinition)
            and (self.test_run.output_path / "benchmark_report.csv").exists()
        )
