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

import json
import logging
from functools import cache
from pathlib import Path

from cloudai.workloads.common.llm_serving import LLMServingBenchReport, LLMServingReportGenerationStrategy, all_gpu_ids

from .vllm import VLLM_BENCH_JSON_FILE, VllmTestDefinition


class VLLMBenchReport(LLMServingBenchReport):
    """Report for vLLM benchmark results."""

    output_throughput: float

    @property
    def throughput(self) -> float:
        return self.output_throughput


@cache
def parse_vllm_bench_output(res_file: Path) -> VLLMBenchReport | None:
    """Parse the vLLM benchmark output file and return a VLLMBenchReport object."""
    if not res_file.is_file():
        return None

    try:
        data = json.loads(res_file.read_text())
        return VLLMBenchReport.model_validate(data)
    except Exception as e:
        logging.debug(f"Error parsing vLLM benchmark output: {e}")
        return None


class VLLMBenchReportGenerationStrategy(LLMServingReportGenerationStrategy[VllmTestDefinition, VLLMBenchReport]):
    """Generate a report for vLLM benchmark results."""

    @property
    def result_file_name(self) -> str:
        return VLLM_BENCH_JSON_FILE

    @property
    def report_title(self) -> str:
        return "vLLM Benchmark Results"

    def parse_output(self, path: Path) -> VLLMBenchReport | None:
        return parse_vllm_bench_output(path)

    def all_gpu_ids(self, tdef: VllmTestDefinition, gpus_per_node: int | None) -> list[int]:
        return all_gpu_ids(tdef, gpus_per_node)
