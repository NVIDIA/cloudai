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

from cloudai.core import TestRun
from cloudai.workloads.vllm import VLLM_BENCH_LOG_FILE, VllmCmdArgs, VllmTestDefinition


class TestVllmSuccessCheck:
    def setup_method(self) -> None:
        self.vllm_tdef = VllmTestDefinition(
            name="vllm",
            description="vLLM benchmark",
            test_template_name="Vllm",
            cmd_args=VllmCmdArgs(docker_image_url="nvcr.io/nvidia/vllm:latest"),
        )

    def test_no_bench_log_file(self, base_tr: TestRun) -> None:
        result = self.vllm_tdef.was_run_successful(base_tr)
        assert not result.is_successful
        assert result.error_message == f"vLLM bench log not found in {base_tr.output_path}."

    def test_successful_job(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        log_file = base_tr.output_path / VLLM_BENCH_LOG_FILE
        log_content = "============ Serving Benchmark Result ============"
        log_file.write_text(log_content)
        result = self.vllm_tdef.was_run_successful(base_tr)
        assert result.is_successful
        assert result.error_message == ""

    def test_failed_job_no_result(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        log_file = base_tr.output_path / VLLM_BENCH_LOG_FILE
        log_content = "Starting benchmark...\nsome line\n"
        log_file.write_text(log_content)
        result = self.vllm_tdef.was_run_successful(base_tr)
        assert not result.is_successful
        assert result.error_message == f"vLLM bench log does not contain benchmark result in {base_tr.output_path}."

    def test_empty_log_file(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        log_file = base_tr.output_path / VLLM_BENCH_LOG_FILE
        log_file.touch()
        result = self.vllm_tdef.was_run_successful(base_tr)
        assert not result.is_successful
        assert result.error_message == f"vLLM bench log does not contain benchmark result in {base_tr.output_path}."
