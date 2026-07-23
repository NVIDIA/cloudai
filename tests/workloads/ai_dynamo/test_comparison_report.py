# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cloudai.core
import cloudai.report_generator.comparison_report
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.ai_dynamo import (
    AIDynamoArgs,
    AIDynamoCmdArgs,
    AIDynamoComparisonReport,
    AIDynamoTestDefinition,
    AIPerf,
    AIPerfAccuracy,
)


def test_ai_dynamo_comparison_report_generates_html(slurm_system: SlurmSystem) -> None:
    test = AIDynamoTestDefinition(
        name="ai-dynamo",
        description="AI Dynamo benchmark",
        test_template_name="AIDynamo",
        cmd_args=AIDynamoCmdArgs(
            docker_image_url="nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest",
            workloads="aiperf.sh",
            dynamo=AIDynamoArgs(),
            aiperf=AIPerf.model_validate({"args": {"concurrency": 4}}),
            aiperf_accuracy=AIPerfAccuracy(
                cli="--model {model} --url {url} --artifact-dir {artifact_dir} --accuracy-benchmark mmlu"
            ),
        ),
    )
    tr = cloudai.core.TestRun(name="ai-dynamo", test=test, num_nodes=2, nodes=[])
    run_dir = slurm_system.output_path / tr.name / "0"
    run_dir.mkdir(parents=True)
    (run_dir / "aiperf_report.csv").write_text(
        "Metric,avg,min,max,p99,p50\n"
        "Inter Token Latency (ms),2.83,2.78,2.91,3.20,2.82\n"
        "Time to First Token (ms),49.87,17.15,99.91,95.00,45.00\n"
        "\n"
        "Metric,Value\n"
        "Output Token Throughput (tokens/sec),595.68\n"
        "Request Count,50\n",
        encoding="utf-8",
    )
    (run_dir / "accuracy_results.csv").write_text(
        "Task,Correct,Total,Accuracy\nOVERALL,35,100,35.00%\n",
        encoding="utf-8",
    )

    report = AIDynamoComparisonReport(
        slurm_system,
        cloudai.core.TestScenario(name="ai-dynamo-comparison", test_runs=[tr]),
        slurm_system.output_path,
        cloudai.report_generator.comparison_report.ComparisonReportConfig(enable=True),
    )
    report.load_test_runs()

    assert len(report.trs) == 1
    result_df = report.extract_data_as_df(report.trs[0])
    values = dict(zip(result_df["metric"], result_df["value"], strict=True))
    assert values["Mean TTFT (ms)"] == 49.87
    assert values["Median TPOT (ms)"] == 2.82
    assert values["Successful Prompts"] == 50.0
    assert values["Throughput"] == 595.68
    assert values["TPS/User"] == 148.92
    assert values["TPS/GPU"] == round(
        595.68 / (2 * (slurm_system.gpus_per_node or 1)),
        4,
    )
    assert values["Accuracy"] == 0.35

    report.generate()

    assert (slurm_system.output_path / "ai_dynamo_comparison.html").exists()
