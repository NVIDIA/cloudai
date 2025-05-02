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
from cloudai.workloads.nemo_run import (
    NeMoRunCmdArgs,
    NeMoRunDataStoreReportGenerationStrategy,
    NeMoRunReportGenerationStrategy,
    NeMoRunTestDefinition,
)


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


@pytest.mark.parametrize(
    "docker_image_url,expected_version",
    [
        ("nvcr.io/nvidia/nemo:24.12.rc3", "24.12.rc3"),
        ("/home/theo/cloudaix/install/nvcr.io_nvidia__nemo__24.12.01.sqsh", "24.12.01"),
        ("custom_docker_image_without_version", "unknown"),
    ],
)
def test_extract_version_from_docker_image(
    slurm_system: SlurmSystem, nemo_tr: TestRun, docker_image_url: str, expected_version: str
) -> None:
    strategy = NeMoRunDataStoreReportGenerationStrategy(slurm_system, nemo_tr)
    version = strategy.extract_version_from_docker_image(docker_image_url)
    assert version == expected_version, (
        f"Expected version '{expected_version}' but got '{version}' for input '{docker_image_url}'"
    )


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
        ("cloudai_llama3_8b_recipe", ("llama3", "8b")),
        ("cloudai_llama3_70b_recipe", ("llama3", "70b")),
        ("cloudai_llama3_405b_recipe", ("llama3", "405b")),
        ("cloudai_nemotron3_8b_recipe", ("nemotron3", "8b")),
        ("cloudai_nemotron4_15b_recipe", ("nemotron4", "15b")),
        ("cloudai_nemotron4_340b_recipe", ("nemotron4", "340b")),
    ],
)
def test_extract_model_info(
    slurm_system: SlurmSystem, nemo_tr: TestRun, input_name: str, expected: Tuple[str, str]
) -> None:
    strategy = NeMoRunDataStoreReportGenerationStrategy(slurm_system, nemo_tr)
    model_info = strategy.extract_model_info(input_name)
    assert model_info == expected, f"Model info extraction failed for {input_name}"


def test_extract_vocab_size(slurm_system: SlurmSystem, nemo_tr_empty_log: TestRun, tmp_path: Path) -> None:
    log_file = tmp_path / "stdout.txt"
    log_file.write_text(
        (
            "[NeMo I 2025-04-14 08:10:56 tokenizer_utils:225] Getting Megatron tokenizer for pretrained model name: "
            "megatron-gpt-345m\n"
            "[NeMo I 2025-04-14 08:10:56 tokenizer_utils:130] Getting HuggingFace AutoTokenizer with "
            "pretrained_model_name: gpt2\n"
            "[NeMo I 2025-04-14 08:11:01 megatron_init:513] Rank 0 has embedding rank: 0\n"
            "[NeMo I 2025-04-14 08:11:05 base:44] Padded vocab_size: 50304, original vocab_size: 50257"
            ", dummy tokens: 47.\n"
            "[NeMo I 2025-04-14 08:11:06 something_else:999] Some other unrelated log line\n"
        )
    )

    strategy = NeMoRunDataStoreReportGenerationStrategy(slurm_system, nemo_tr_empty_log)
    vocab_size = strategy.extract_vocab_size(log_file)

    assert vocab_size == 50304, f"Expected vocab size 50304, but got {vocab_size}"


@pytest.mark.parametrize(
    "lines,expected_config",
    [
        (
            [
                "#!/bin/bash",
                "echo Hello World",
                "srun python train.py -y trainer.max_steps=100 data.seq_length=2048",
            ],
            "trainer.max_steps=100 data.seq_length=2048",
        ),
        (
            [
                "#SBATCH --job-name=test",
                "srun echo nothing here",
                "srun python run.py --config something -y data.global_batch_size=256",
            ],
            "data.global_batch_size=256",
        ),
        (
            [
                "# just comments",
                "srun python script.py",  # no `-y`
            ],
            "",
        ),
        (
            [
                "   ",  # empty line
                "srun --mpi=pmix python app.py -y trainer.num_nodes=2",
            ],
            "trainer.num_nodes=2",
        ),
        (
            [],  # empty file
            "",
        ),
    ],
)
def test_extract_base_config_from_sbatch_script_parametrized(
    slurm_system: SlurmSystem, nemo_tr: TestRun, lines: list[str], expected_config: str
) -> None:
    strategy = NeMoRunDataStoreReportGenerationStrategy(slurm_system, nemo_tr)
    sbatch_path = nemo_tr.output_path / "cloudai_sbatch_script.sh"
    sbatch_path.write_text("\n".join(lines), encoding="utf-8")

    extracted = strategy.extract_base_config_from_sbatch_script(nemo_tr.output_path)
    assert extracted == expected_config, f"Expected: {expected_config}, Got: {extracted}"
