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

from cloudai.workloads.common.llm_serving import LLMServingReportGenerationStrategy

from .sglang import SGLANG_BENCH_JSONL_FILE, SGLangBenchReport, SglangTestDefinition, parse_sglang_bench_output
from .slurm_command_gen_strategy import sglang_all_gpu_ids


class SGLangBenchReportGenerationStrategy(LLMServingReportGenerationStrategy[SglangTestDefinition, SGLangBenchReport]):
    """Generate report and metrics for SGLang benchmark output."""

    @property
    def result_file_name(self) -> str:
        return SGLANG_BENCH_JSONL_FILE

    @property
    def report_title(self) -> str:
        return "SGLang Benchmark Results"

    def parse_output(self, path: Path) -> SGLangBenchReport | None:
        return parse_sglang_bench_output(path)

    def all_gpu_ids(self, tdef: SglangTestDefinition, gpus_per_node: int | None) -> list[int]:
        return sglang_all_gpu_ids(tdef, gpus_per_node)
