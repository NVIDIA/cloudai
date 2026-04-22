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
from pathlib import Path

from cloudai import TestRun, TestScenario
from cloudai.report_generator.comparison_report import ComparisonReportConfig
from cloudai.report_generator.groups import TestRunsGrouper
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.sglang import (
    SglangBenchCmdArgs,
    SglangCmdArgs,
    SGLangComparisonReport,
    SglangTestDefinition,
)
from cloudai.workloads.sglang.sglang import SGLANG_BENCH_JSONL_FILE
from cloudai.workloads.vllm import (
    VllmBenchCmdArgs,
    VllmCmdArgs,
    VLLMComparisonReport,
    VllmTestDefinition,
)
from cloudai.workloads.vllm.vllm import VLLM_BENCH_JSON_FILE


def _write_result(run_dir: Path, file_name: str, content: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / file_name).write_text(content, encoding="utf-8")


def test_vllm_comparison_report_generates_html(slurm_system: SlurmSystem) -> None:
    tr1 = TestRun(
        name="vllm-8",
        test=VllmTestDefinition(
            name="vllm-8",
            description="vLLM benchmark",
            test_template_name="vllm",
            cmd_args=VllmCmdArgs(docker_image_url="nvcr.io/nvidia/vllm:latest", model="Qwen/Qwen3-0.6B"),
            bench_cmd_args=VllmBenchCmdArgs(max_concurrency=8),
        ),
        num_nodes=1,
        nodes=[],
    )
    tr2 = TestRun(
        name="vllm-16",
        test=VllmTestDefinition(
            name="vllm-16",
            description="vLLM benchmark",
            test_template_name="vllm",
            cmd_args=VllmCmdArgs(docker_image_url="nvcr.io/nvidia/vllm:latest", model="Qwen/Qwen3-0.6B"),
            bench_cmd_args=VllmBenchCmdArgs(max_concurrency=16),
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
                "completed": 30,
                "mean_ttft_ms": 120.0,
                "median_ttft_ms": 118.0,
                "p99_ttft_ms": 200.0,
                "mean_tpot_ms": 12.0,
                "median_tpot_ms": 11.0,
                "p99_tpot_ms": 19.0,
                "output_throughput": 2400.0,
                "max_concurrency": 8,
            }
        ),
    )
    _write_result(
        slurm_system.output_path / tr2.name / "0",
        VLLM_BENCH_JSON_FILE,
        json.dumps(
            {
                "num_prompts": 30,
                "completed": 28,
                "mean_ttft_ms": 115.0,
                "median_ttft_ms": 110.0,
                "p99_ttft_ms": 180.0,
                "mean_tpot_ms": 10.0,
                "median_tpot_ms": 9.0,
                "p99_tpot_ms": 16.0,
                "output_throughput": 2600.0,
                "max_concurrency": 16,
            }
        ),
    )

    report = VLLMComparisonReport(
        slurm_system,
        TestScenario(name="vllm-comparison", test_runs=[tr1, tr2]),
        slurm_system.output_path,
        ComparisonReportConfig(enable=True, group_by=["model"]),
    )

    report.load_test_runs()
    groups = TestRunsGrouper(report.trs, report.group_by).groups()
    tables = report.create_tables(groups)
    assert len(tables) == 2
    assert "bench_cmd_args.max_concurrency=8" in tables[0].columns[1].header  # type: ignore

    df = report.extract_data_as_df(report.trs[0])
    throughput_row = df[df["metric"] == "TPS/GPU"].iloc[0]
    assert throughput_row["value"] == 300.0

    report.generate()
    assert (slurm_system.output_path / "vllm_comparison.html").exists()


def test_sglang_comparison_report_generates_html(slurm_system: SlurmSystem) -> None:
    tr1 = TestRun(
        name="sglang-8",
        test=SglangTestDefinition(
            name="sglang-8",
            description="SGLang benchmark",
            test_template_name="sglang",
            cmd_args=SglangCmdArgs(docker_image_url="docker.io/lmsysorg/sglang:dev", model="Qwen/Qwen3-8B"),
            bench_cmd_args=SglangBenchCmdArgs(max_concurrency=8),
        ),
        num_nodes=1,
        nodes=[],
    )
    tr2 = TestRun(
        name="sglang-16",
        test=SglangTestDefinition(
            name="sglang-16",
            description="SGLang benchmark",
            test_template_name="sglang",
            cmd_args=SglangCmdArgs(docker_image_url="docker.io/lmsysorg/sglang:dev", model="Qwen/Qwen3-8B"),
            bench_cmd_args=SglangBenchCmdArgs(max_concurrency=16),
        ),
        num_nodes=1,
        nodes=[],
    )
    _write_result(
        slurm_system.output_path / tr1.name / "0",
        SGLANG_BENCH_JSONL_FILE,
        json.dumps(
            {
                "num_prompts": 30,
                "completed": 30,
                "request_throughput": 2400.0,
                "max_concurrency": 8,
                "mean_ttft_ms": 120.0,
                "median_ttft_ms": 100.0,
                "p99_ttft_ms": 200.0,
                "mean_tpot_ms": 12.0,
                "median_tpot_ms": 10.0,
                "p99_tpot_ms": 20.0,
            }
        )
        + "\n",
    )
    _write_result(
        slurm_system.output_path / tr2.name / "0",
        SGLANG_BENCH_JSONL_FILE,
        json.dumps(
            {
                "num_prompts": 30,
                "completed": 29,
                "request_throughput": 2600.0,
                "max_concurrency": 16,
                "mean_ttft_ms": 115.0,
                "median_ttft_ms": 95.0,
                "p99_ttft_ms": 190.0,
                "mean_tpot_ms": 11.0,
                "median_tpot_ms": 9.0,
                "p99_tpot_ms": 18.0,
            }
        )
        + "\n",
    )

    report = SGLangComparisonReport(
        slurm_system,
        TestScenario(name="sglang-comparison", test_runs=[tr1, tr2]),
        slurm_system.output_path,
        ComparisonReportConfig(enable=True, group_by=["model"]),
    )

    report.load_test_runs()
    groups = TestRunsGrouper(report.trs, report.group_by).groups()
    tables = report.create_tables(groups)
    assert len(tables) == 2
    assert "bench_cmd_args.max_concurrency=16" in tables[0].columns[2].header  # type: ignore

    df = report.extract_data_as_df(report.trs[1])
    throughput_row = df[df["metric"] == "TPS/User"].iloc[0]
    assert throughput_row["value"] == 162.5

    report.generate()
    assert (slurm_system.output_path / "sglang_comparison.html").exists()
