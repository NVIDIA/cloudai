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
import pathlib

import cloudai.core
import cloudai.report_generator.comparison_report
import cloudai.report_generator.groups
import cloudai.systems.slurm
from cloudai.workloads.vllm import (
    VLLM_BENCH_JSON_FILE,
    VLLM_GSM8K_JSON_FILE,
    VllmBenchCmdArgs,
    VllmCmdArgs,
    VLLMComparisonReport,
    VllmSemanticEvalCmdArgs,
    VllmTestDefinition,
)


def _write_result(run_dir: pathlib.Path, file_name: str, content: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / file_name).write_text(content, encoding="utf-8")


def test_llm_comparison_report_generates_html(slurm_system: cloudai.systems.slurm.SlurmSystem) -> None:
    tr1 = cloudai.core.TestRun(
        name="vllm-8",
        test=VllmTestDefinition(
            name="vllm",
            description="vLLM benchmark",
            test_template_name="vllm",
            cmd_args=VllmCmdArgs(docker_image_url="nvcr.io/nvidia/vllm:latest", model="Qwen/Qwen3-0.6B"),
            bench_cmd_args=VllmBenchCmdArgs(max_concurrency=8),
            semantic_eval_cmd_args=VllmSemanticEvalCmdArgs(),
        ),
        num_nodes=1,
        nodes=[],
    )
    tr2 = cloudai.core.TestRun(
        name="vllm-16",
        test=VllmTestDefinition(
            name="vllm",
            description="vLLM benchmark",
            test_template_name="vllm",
            cmd_args=VllmCmdArgs(docker_image_url="nvcr.io/nvidia/vllm:latest", model="Qwen/Qwen3-0.6B"),
            bench_cmd_args=VllmBenchCmdArgs(max_concurrency=16),
            semantic_eval_cmd_args=VllmSemanticEvalCmdArgs(),
        ),
        num_nodes=1,
        nodes=[],
    )
    _write_result(
        slurm_system.output_path / tr1.name / "0",
        VLLM_BENCH_JSON_FILE,
        json.dumps(
            {
                "num_prompts": 30,
                "completed": 27,
                "mean_ttft_ms": 100.0,
                "median_ttft_ms": 90.0,
                "p99_ttft_ms": 150.0,
                "mean_tpot_ms": 10.0,
                "median_tpot_ms": 9.0,
                "p99_tpot_ms": 15.0,
                "output_throughput": 1200.0,
                "max_concurrency": 8,
            }
        ),
    )
    _write_result(slurm_system.output_path / tr1.name / "0", VLLM_GSM8K_JSON_FILE, '{"accuracy": 0.81}')
    _write_result(
        slurm_system.output_path / tr2.name / "0",
        VLLM_BENCH_JSON_FILE,
        json.dumps(
            {
                "num_prompts": 30,
                "completed": 30,
                "mean_ttft_ms": 80.0,
                "median_ttft_ms": 70.0,
                "p99_ttft_ms": 130.0,
                "mean_tpot_ms": 8.0,
                "median_tpot_ms": 7.0,
                "p99_tpot_ms": 13.0,
                "output_throughput": 1800.0,
                "max_concurrency": 16,
            }
        ),
    )
    _write_result(slurm_system.output_path / tr2.name / "0", VLLM_GSM8K_JSON_FILE, '{"accuracy": 0.9}')

    report = VLLMComparisonReport(
        slurm_system,
        cloudai.core.TestScenario(name="vllm-comparison", test_runs=[tr1, tr2]),
        slurm_system.output_path,
        cloudai.report_generator.comparison_report.ComparisonReportConfig(enable=True, group_by=[]),
    )

    report.load_test_runs()
    assert len(report.trs) == 2
    tables = report.create_tables(
        cloudai.report_generator.groups.TestRunsGrouper(report.trs, [], report.comparison_values).groups()
    )
    latency_table = tables[0]
    success_table = tables[1]
    throughput_table = tables[2]
    quality_table = tables[3]

    assert "bench_cmd_args.max_concurrency=8" in str(latency_table.columns[1].header)
    assert "bench_cmd_args.max_concurrency=16" in str(latency_table.columns[2].header)
    assert list(latency_table.columns[0].cells)[:3] == ["Mean TTFT (ms)", "Median TTFT (ms)", "P99 TTFT (ms)"]
    assert list(success_table.columns[0].cells) == ["Successful Prompts", "Successful Prompts (%)"]
    assert list(success_table.columns[3].cells) == [
        cloudai.report_generator.comparison_report.ComparisonReport._format_diff_cell(27.0, 30.0),
        cloudai.report_generator.comparison_report.ComparisonReport._format_diff_cell(90.0, 100.0),
    ]
    throughput_row = list(throughput_table.columns[0].cells).index("TPS/GPU")
    assert list(throughput_table.columns[1].cells)[throughput_row] == "150.0"
    assert list(quality_table.columns[0].cells) == ["Accuracy"]
    assert list(quality_table.columns[1].cells) == ["0.81"]
    assert list(quality_table.columns[2].cells) == ["0.9"]

    report.generate()

    assert (slurm_system.output_path / "vllm_comparison.html").exists()
