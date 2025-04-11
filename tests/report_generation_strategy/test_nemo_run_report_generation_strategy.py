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

from pathlib import Path
from typing import Tuple
from unittest.mock import Mock

import pytest

from cloudai import Test, TestRun
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nemo_run import NeMoRunCmdArgs, NeMoRunReportGenerationStrategy, NeMoRunTestDefinition


@pytest.fixture
def nemo_tr(tmp_path: Path) -> TestRun:
    test = Test(
        test_definition=NeMoRunTestDefinition(
            name="nemo",
            description="desc",
            test_template_name="t",
            cmd_args=NeMoRunCmdArgs(docker_image_url="docker://url", task="task", recipe_name="recipe"),
        ),
        test_template=Mock(),
    )
    tr = TestRun(name="nemo", test=test, num_nodes=1, nodes=[], output_path=tmp_path)
    tr.reports = {NeMoRunReportGenerationStrategy}

    stdout_content = (
        "Training epoch 0, iteration 17/99 | lr: 2.699e-06 | global_batch_size: 128 | global_step: 17 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.64 | consumed_samples: 2304\n"
        "Training epoch 0, iteration 18/99 | lr: 2.849e-06 | global_batch_size: 128 | global_step: 18 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.64 | consumed_samples: 2432\n"
        "Training epoch 0, iteration 19/99 | lr: 2.999e-06 | global_batch_size: 128 | global_step: 19 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.64 | consumed_samples: 2560\n"
        "Training epoch 0, iteration 20/99 | lr: 3.148e-06 | global_batch_size: 128 | global_step: 20 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.65 | consumed_samples: 2688\n"
        "Training epoch 0, iteration 21/99 | lr: 3.298e-06 | global_batch_size: 128 | global_step: 21 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.87 | consumed_samples: 2816\n"
        "Training epoch 0, iteration 22/99 | lr: 3.448e-06 | global_batch_size: 128 | global_step: 22 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.87 | consumed_samples: 2944\n"
        "Training epoch 0, iteration 23/99 | lr: 3.598e-06 | global_batch_size: 128 | global_step: 23 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.63 | consumed_samples: 3072\n"
        "Training epoch 0, iteration 24/99 | lr: 3.748e-06 | global_batch_size: 128 | global_step: 24 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 13.04 | consumed_samples: 3200\n"
        "Training epoch 0, iteration 25/99 | lr: 3.898e-06 | global_batch_size: 128 | global_step: 25 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.64 | consumed_samples: 3328\n"
        "Training epoch 0, iteration 26/99 | lr: 4.048e-06 | global_batch_size: 128 | global_step: 26 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.65 | consumed_samples: 3456\n"
        "Training epoch 0, iteration 27/99 | lr: 4.198e-06 | global_batch_size: 128 | global_step: 27 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.65 | consumed_samples: 3584\n"
        "Training epoch 0, iteration 28/99 | lr: 4.348e-06 | global_batch_size: 128 | global_step: 28 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.65 | consumed_samples: 3712\n"
    )

    (tr.output_path / "stdout.txt").write_text(stdout_content)

    return tr


@pytest.fixture
def nemo_tr_empty_log(tmp_path: Path) -> TestRun:
    test = Test(
        test_definition=NeMoRunTestDefinition(
            name="nemo",
            description="desc",
            test_template_name="template",
            cmd_args=NeMoRunCmdArgs(
                docker_image_url="docker://url",
                task="task",
                recipe_name="recipe",
            ),
        ),
        test_template=Mock(),
    )
    return TestRun(name="nemo", test=test, num_nodes=1, nodes=[], output_path=tmp_path)


@pytest.fixture
def nemo_tr_encoded(tmp_path: Path) -> TestRun:
    test = Test(
        test_definition=NeMoRunTestDefinition(
            name="nemo",
            description="desc",
            test_template_name="t",
            cmd_args=NeMoRunCmdArgs(docker_image_url="docker://url", task="task", recipe_name="recipe"),
        ),
        test_template=Mock(),
    )
    tr = TestRun(name="nemo", test=test, num_nodes=1, nodes=[], output_path=tmp_path)

    stdout_content = (
        "┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        "┃[1;35m [0m[1;35mArgument Name   [0m[1;35m [0m┃[1;35m [0m[1;35mResolved Value"
        "┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩\n"
        "│[2m [0m[2mdata            [0m[2m [0m│ [1;35mMockDataModule[0m[1m([0m[33mseq_length["
        "Training epoch 0, iteration 17/99 | lr: 2.699e-06 | global_batch_size: 128 | global_step: 17 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.64 | consumed_samples: 2304\n"
        "Training epoch 0, iteration 18/99 | lr: 2.849e-06 | global_batch_size: 128 | global_step: 18 | "
        "reduced_train_loss: 11.03 | train_step_timing in s: 12.64 | consumed_samples: 2432\n"
        "Training epoch 0, iteration 19/99 | lr: 2.999e-06 | global_batch_size: 128 | global_step: 19 | "
    )

    (tr.output_path / "stdout.txt").write_text(stdout_content)

    return tr


def test_nemo_can_handle_directory(slurm_system: SlurmSystem, nemo_tr: TestRun) -> None:
    strategy = NeMoRunReportGenerationStrategy(slurm_system, nemo_tr)
    assert strategy.can_handle_directory()


def test_nemo_generate_report(slurm_system: SlurmSystem, nemo_tr: TestRun) -> None:
    strategy = NeMoRunReportGenerationStrategy(slurm_system, nemo_tr)
    strategy.generate_report()

    summary_file = nemo_tr.output_path / "report.txt"
    assert summary_file.is_file(), "Summary report was not generated."

    summary_content = summary_file.read_text().strip().split("\n")
    assert len(summary_content) == 4, "Summary file should contain four lines (avg, median, min, max)."

    expected_values = {
        "Average": 12.74,
        "Median": 12.65,
        "Min": 12.63,
        "Max": 13.04,
    }

    for line in summary_content:
        key, value = line.split(": ")
        assert pytest.approx(float(value), 0.01) == expected_values[key], f"{key} value mismatch."


def test_nemo_generate_report_encoded(slurm_system: SlurmSystem, nemo_tr_encoded: TestRun) -> None:
    strategy = NeMoRunReportGenerationStrategy(slurm_system, nemo_tr_encoded)
    strategy.generate_report()

    summary_file = nemo_tr_encoded.output_path / "report.txt"
    assert summary_file.is_file(), "Summary report was not generated."

    summary_content = summary_file.read_text(encoding="utf-8", errors="ignore").strip().split("\n")
    assert len(summary_content) == 4, "Summary file should contain four lines (avg, median, min, max)."

    expected_values = {
        "Average": 12.74,
        "Median": 12.65,
        "Min": 12.63,
        "Max": 12.64,
    }

    for line in summary_content:
        key, value = line.split(": ")
        assert pytest.approx(float(value), 0.01) == expected_values[key], f"{key} value mismatch."


@pytest.mark.parametrize("metric", ["default", "step-time"])
def test_metrics(nemo_tr: TestRun, slurm_system: SlurmSystem, metric: str):
    nemo_tr.test.test_definition.agent_metric = metric
    value = nemo_tr.get_metric_value(slurm_system)
    assert value == 12.714166666666666


def test_extract_timings_valid_file(slurm_system: SlurmSystem, nemo_tr_empty_log: TestRun) -> None:
    stdout_file = nemo_tr_empty_log.output_path / "stdout.txt"
    stdout_file.write_text(
        "Training epoch 0, iteration 17/99 | train_step_timing in s: 12.64 | global_step: 17\n"
        "Training epoch 0, iteration 18/99 | train_step_timing in s: 12.65 | global_step: 18\n"
        "Training epoch 0, iteration 19/99 | train_step_timing in s: 12.66 | global_step: 19\n"
    )
    strategy = NeMoRunReportGenerationStrategy(slurm_system, nemo_tr_empty_log)
    timings = strategy._parse_timings(stdout_file)
    assert timings == [12.64, 12.65, 12.66]


def test_extract_timings_missing_file(slurm_system: SlurmSystem, nemo_tr_empty_log: TestRun, tmp_path: Path) -> None:
    stdout_file = tmp_path / "missing_stdout.txt"
    strategy = NeMoRunReportGenerationStrategy(slurm_system, nemo_tr_empty_log)
    timings = strategy._parse_timings(stdout_file)
    assert timings == [], "Timings extraction should return an empty list for missing file."


def test_extract_timings_invalid_content(slurm_system: SlurmSystem, nemo_tr_empty_log: TestRun, tmp_path: Path) -> None:
    stdout_file = tmp_path / "stdout.txt"
    stdout_file.write_text("Invalid content without timing information\n")
    strategy = NeMoRunReportGenerationStrategy(slurm_system, nemo_tr_empty_log)
    timings = strategy._parse_timings(stdout_file)
    assert timings == [], "Timings extraction should return an empty list for invalid content."


def test_extract_timings_file_not_found(slurm_system: SlurmSystem, nemo_tr_empty_log: TestRun, tmp_path: Path) -> None:
    stdout_file = tmp_path / "nonexistent_stdout.txt"
    strategy = NeMoRunReportGenerationStrategy(slurm_system, nemo_tr_empty_log)
    timings = strategy._parse_timings(stdout_file)
    assert timings == [], "Timings extraction should return an empty list when the file does not exist."


def test_generate_report_no_timings(slurm_system: SlurmSystem, nemo_tr: TestRun, tmp_path: Path) -> None:
    nemo_tr.output_path = tmp_path
    stdout_file = nemo_tr.output_path / "stdout.txt"
    stdout_file.write_text("No valid timing information\n")

    strategy = NeMoRunReportGenerationStrategy(slurm_system, nemo_tr)
    strategy.generate_report()

    summary_file = nemo_tr.output_path / "report.txt"
    assert not summary_file.exists(), "Report should not be generated if no valid timings are found."


def test_generate_report_partial_timings(slurm_system: SlurmSystem, nemo_tr: TestRun, tmp_path: Path) -> None:
    nemo_tr.output_path = tmp_path
    stdout_file = nemo_tr.output_path / "stdout.txt"
    stdout_file.write_text(
        "Training epoch 0, iteration 17/99 | train_step_timing in s: 12.64 | global_step: 17\n"
        "Invalid line without timing\n"
        "Training epoch 0, iteration 18/99 | train_step_timing in s: 12.65 | global_step: 18\n"
    )

    strategy = NeMoRunReportGenerationStrategy(slurm_system, nemo_tr)
    strategy.generate_report()

    summary_file = nemo_tr.output_path / "report.txt"
    assert summary_file.is_file(), "Report should be generated even with partial valid timings."

    summary_content = summary_file.read_text().strip().split("\n")
    assert len(summary_content) == 4, "Summary file should contain four lines (avg, median, min, max)."

    expected_values = {
        "Average": 12.645,
        "Median": 12.645,
        "Min": 12.64,
        "Max": 12.65,
    }

    for line in summary_content:
        key, value = line.split(": ")
        assert pytest.approx(float(value), 0.01) == expected_values[key], f"{key} value mismatch."


@pytest.mark.parametrize(
    "input_name,expected",
    [
        ("nemotron4_15b_64k", ("nemotron4", "15b")),
        ("nemotron3_22b", ("nemotron3", "22b")),
        ("baichuan2_7b", ("baichuan2", "7b")),
        ("hyena_1b", ("hyena", "1b")),
        ("qwen25_14b", ("qwen25", "14b")),
        ("nemotron5_hybrid_47b", ("nemotron5_hybrid", "47b")),
        ("t5_3b", ("t5", "3b")),
        ("hyena_base", ("hyena_base", "")),
        ("llama31_nemotron_nano_8b", ("llama31_nemotron_nano", "8b")),
        ("starcoder2_3b", ("starcoder2", "3b")),
        ("nemotron4_15b", ("nemotron4", "15b")),
        ("llama3_8b_64k", ("llama3", "8b")),
        ("llama3_8b_128k", ("llama3", "8b")),
        ("nemotron3_22b_16k", ("nemotron3", "22b")),
        ("hyena_40b", ("hyena", "40b")),
        ("t5_11b", ("t5", "11b")),
        ("starcoder2_7b", ("starcoder2", "7b")),
        ("hyena_7b", ("hyena", "7b")),
        ("llama31_8b", ("llama31", "8b")),
        ("gemma2_9b", ("gemma2", "9b")),
        ("nemotron5_hybrid_56b", ("nemotron5_hybrid", "56b")),
        ("t5_220m", ("t5", "220m")),
        ("nemotron3_4b", ("nemotron3", "4b")),
        ("qwen2_1p5b", ("qwen2", "1p5b")),
        ("bert_340m", ("bert", "340m")),
        ("mamba2_8b", ("mamba2", "8b")),
        ("deepseek_v2_lite", ("deepseek_v2_lite", "")),
        ("qwen2_7b", ("qwen2", "7b")),
        ("nemotron3_22b_64k", ("nemotron3", "22b")),
        ("llama3_8b_16k", ("llama3", "8b")),
        ("starcoder_15b", ("starcoder", "15b")),
        ("nemotron4_340b", ("nemotron4", "340b")),
        ("starcoder2_15b", ("starcoder2", "15b")),
        ("llama3_8b", ("llama3", "8b")),
        ("llama31_70b", ("llama31", "70b")),
        ("phi3_mini_4k_instruct", ("phi3_mini_4k_instruct", "")),
        ("nemotron4_15b_16k", ("nemotron4", "15b")),
        ("qwen25_500m", ("qwen25", "500m")),
        ("qwen25_72b", ("qwen25", "72b")),
        ("mamba2_370m", ("mamba2", "370m")),
        ("gemma_2b", ("gemma", "2b")),
        ("llama3_70b", ("llama3", "70b")),
        ("nemotron5_hybrid_8b", ("nemotron5_hybrid", "8b")),
        ("hf_auto_model_for_causal_lm", ("hf_auto_model_for_causal_lm", "")),
        ("gemma2_27b", ("gemma2", "27b")),
        ("chatglm3_6b", ("chatglm3", "6b")),
        ("mixtral_8x7b_64k", ("mixtral_8x7b_64k", "")),
        ("qwen25_7b", ("qwen25", "7b")),
        ("mamba2_1_3b", ("mamba2_1", "3b")),
        ("mamba2_780m", ("mamba2", "780m")),
        ("llama3_70b_64k", ("llama3", "70b")),
        ("llama32_3b", ("llama32", "3b")),
        ("mixtral_8x22b_64k", ("mixtral_8x22b_64k", "")),
        ("nemotron3_8b", ("nemotron3", "8b")),
        ("gemma_7b", ("gemma", "7b")),
        ("llama32_1b", ("llama32", "1b")),
        ("my_new_model_12b", ("my_new_model", "12b")),
        ("mistral_7b", ("mistral", "7b")),
        ("e5_340m", ("e5", "340m")),
        ("llama31_405b", ("llama31", "405b")),
        ("gpt3_175b", ("gpt3", "175b")),
        ("qwen25_1p5b", ("qwen25", "1p5b")),
        ("mistral_nemo_12b", ("mistral_nemo", "12b")),
        ("mixtral_8x7b", ("mixtral_8x7b", "")),
        ("llama3_70b_16k", ("llama3", "70b")),
        ("mamba2_2_7b", ("mamba2_2", "7b")),
        ("deepseek_v3", ("deepseek_v3", "")),
        ("llama2_7b", ("llama2", "7b")),
        ("mixtral_8x22b", ("mixtral_8x22b", "")),
        ("mamba2_hybrid_8b", ("mamba2_hybrid", "8b")),
        ("qwen2_72b", ("qwen2", "72b")),
        ("qwen25_32b", ("qwen25", "32b")),
        ("gemma2_2b", ("gemma2", "2b")),
        ("deepseek_v2", ("deepseek_v2", "")),
        ("mixtral_8x7b_16k", ("mixtral_8x7b_16k", "")),
        ("llama31_nemotron_70b", ("llama31_nemotron", "70b")),
        ("nvembed_llama_1b", ("nvembed_llama", "1b")),
        ("qwen2_500m", ("qwen2", "500m")),
        ("mamba2_130m", ("mamba2", "130m")),
        ("bert_110m", ("bert", "110m")),
    ],
)
def test_extract_model_info(
    slurm_system: SlurmSystem, nemo_tr: TestRun, input_name: str, expected: Tuple[str, str]
) -> None:
    strategy = NeMoRunReportGenerationStrategy(slurm_system, nemo_tr)
    model_info = strategy.extract_model_info(input_name)
    assert model_info == expected, f"Model info extraction failed for {input_name}"
