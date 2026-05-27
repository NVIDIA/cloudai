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

from cloudai.workloads.common.llm_serving import (
    SGLANG_SEMANTIC_EVAL_LOG_FILE,
    VLLM_GSM8K_JSON_FILE,
    VLLM_SEMANTIC_EVAL_LOG_FILE,
    parse_sglang_semantic_accuracy,
    parse_vllm_semantic_accuracy,
)


def test_parse_vllm_semantic_accuracy_from_json(tmp_path: Path) -> None:
    (tmp_path / VLLM_GSM8K_JSON_FILE).write_text('{"accuracy": 0.91}', encoding="utf-8")

    assert parse_vllm_semantic_accuracy(tmp_path) == 0.91


def test_parse_vllm_semantic_accuracy_falls_back_to_log(tmp_path: Path) -> None:
    (tmp_path / VLLM_GSM8K_JSON_FILE).write_text("{invalid", encoding="utf-8")
    (tmp_path / VLLM_SEMANTIC_EVAL_LOG_FILE).write_text("Accuracy: 0.742\n", encoding="utf-8")

    assert parse_vllm_semantic_accuracy(tmp_path) == 0.742


def test_parse_vllm_semantic_accuracy_missing_or_invalid(tmp_path: Path) -> None:
    (tmp_path / VLLM_SEMANTIC_EVAL_LOG_FILE).write_text("no accuracy here\n", encoding="utf-8")

    assert parse_vllm_semantic_accuracy(tmp_path) is None


def test_parse_sglang_semantic_accuracy_from_score(tmp_path: Path) -> None:
    log_path = tmp_path / SGLANG_SEMANTIC_EVAL_LOG_FILE
    log_path.write_text("Total latency: 1.000 s\nScore: 0.812\n", encoding="utf-8")

    assert parse_sglang_semantic_accuracy(tmp_path) == 0.812


def test_parse_sglang_semantic_accuracy_from_legacy_accuracy(tmp_path: Path) -> None:
    log_path = tmp_path / SGLANG_SEMANTIC_EVAL_LOG_FILE
    log_path.write_text("Accuracy: 0.945\nInvalid: 0.000\n", encoding="utf-8")

    assert parse_sglang_semantic_accuracy(tmp_path) == 0.945


def test_parse_sglang_semantic_accuracy_missing_or_invalid(tmp_path: Path) -> None:
    missing_output_path = tmp_path / "missing"
    missing_output_path.mkdir()
    log_path = tmp_path / SGLANG_SEMANTIC_EVAL_LOG_FILE
    log_path.write_text("no score here\n", encoding="utf-8")

    assert parse_sglang_semantic_accuracy(missing_output_path) is None
    assert parse_sglang_semantic_accuracy(tmp_path) is None
