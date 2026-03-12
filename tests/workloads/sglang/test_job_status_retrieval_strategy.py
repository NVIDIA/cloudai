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

from cloudai.core import TestRun
from cloudai.workloads.sglang import SGLANG_BENCH_JSONL_FILE, SglangCmdArgs, SglangTestDefinition


class TestSglangSuccessCheck:
    def setup_method(self) -> None:
        self.sglang_tdef = SglangTestDefinition(
            name="sglang",
            description="SGLang benchmark",
            test_template_name="sglang",
            cmd_args=SglangCmdArgs(docker_image_url="docker.io/lmsysorg/sglang:dev"),
        )

    def test_no_bench_log_file(self, base_tr: TestRun) -> None:
        result = self.sglang_tdef.was_run_successful(base_tr)
        assert not result.is_successful
        assert result.error_message == f"SGLang bench jsonl not found in {base_tr.output_path}."

    def test_successful_job(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        log_file = base_tr.output_path / SGLANG_BENCH_JSONL_FILE
        log_file.write_text(json.dumps({"completed": 3}) + "\n")

        result = self.sglang_tdef.was_run_successful(base_tr)

        assert result.is_successful
        assert result.error_message == ""

    def test_failed_job_no_successful_requests(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        log_file = base_tr.output_path / SGLANG_BENCH_JSONL_FILE
        log_file.write_text(json.dumps({"completed": 0}) + "\n")

        result = self.sglang_tdef.was_run_successful(base_tr)

        assert not result.is_successful
        assert (
            result.error_message == f"SGLang bench jsonl does not contain successful requests in {base_tr.output_path}."
        )
