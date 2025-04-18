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

from cloudai.workloads.nemo_run import NeMoRunJobStatusRetrievalStrategy


class TestNeMoRunJobStatusRetrievalStrategy:
    def setup_method(self) -> None:
        self.js = NeMoRunJobStatusRetrievalStrategy()

    def test_no_stderr_file(self, tmp_path: Path) -> None:
        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            f"stderr.txt file not found in the specified output directory {tmp_path}. "
            "This file is expected to be created as part of the NeMo training job. "
            "Please ensure the job was submitted and executed properly. "
            f"You can try re-running the job manually and verify that {tmp_path / 'stderr.txt'} is created "
            "with the expected output. If the issue persists, contact the system administrator."
        )

    def test_successful_job(self, tmp_path: Path) -> None:
        stderr_file = tmp_path / "stderr.txt"
        stderr_content = """
        [NeMo Train] Trainer.fit` stopped: `max_steps=100` reached.
        """
        stderr_file.write_text(stderr_content)
        result = self.js.get_job_status(tmp_path)
        assert result.is_successful
        assert result.error_message == ""

    def test_missing_max_steps_indicator(self, tmp_path: Path) -> None:
        stderr_file = tmp_path / "stderr.txt"
        stderr_content = """
        [NeMo Train] Trainer.fit` stopped: training completed successfully.
        """
        stderr_file.write_text(stderr_content)
        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert "'max_steps='" in result.error_message
        assert "'reached'" in result.error_message

    def test_missing_reached_indicator(self, tmp_path: Path) -> None:
        stderr_file = tmp_path / "stderr.txt"
        stderr_content = """
        [NeMo Train] Trainer.fit` stopped: `max_steps=100`.
        """
        stderr_file.write_text(stderr_content)
        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert "'reached'" in result.error_message
        assert "'max_steps='" not in result.error_message

    def test_missing_both_indicators(self, tmp_path: Path) -> None:
        stderr_file = tmp_path / "stderr.txt"
        stderr_content = """
        [NeMo Train] Trainer.fit stopped unexpectedly.
        """
        stderr_file.write_text(stderr_content)
        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert "'max_steps='" in result.error_message
        assert "'reached'" in result.error_message
