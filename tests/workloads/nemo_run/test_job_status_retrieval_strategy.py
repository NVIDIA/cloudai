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

from cloudai.core import TestRun
from cloudai.workloads.nemo_run import NeMoRunCmdArgs, NeMoRunTestDefinition


class TestNeMoRunSuccessCheck:
    def setup_method(self) -> None:
        self.nemorun_tdef = NeMoRunTestDefinition(
            name="n",
            description="d",
            test_template_name="tt",
            cmd_args=NeMoRunCmdArgs(docker_image_url="", task="", recipe_name=""),
        )

    def test_no_stderr_file(self, base_tr: TestRun) -> None:
        result = self.nemorun_tdef.was_run_successful(base_tr)
        assert not result.is_successful
        assert result.error_message == (
            f"stderr.txt file not found in the specified output directory {base_tr.output_path}. "
            "This file is expected to be created as part of the NeMo training job. "
            "Please ensure the job was submitted and executed properly. "
            f"You can try re-running the job manually and verify that {base_tr.output_path / 'stderr.txt'} is created "
            "with the expected output. If the issue persists, contact the system administrator."
        )

    def test_successful_job(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        stderr_file = base_tr.output_path / "stderr.txt"
        stderr_content = """
        [NeMo Train] Trainer.fit` stopped: `max_steps=100` reached.
        """
        stderr_file.write_text(stderr_content)
        result = self.nemorun_tdef.was_run_successful(base_tr)
        assert result.is_successful
        assert result.error_message == ""

    def test_missing_max_steps_indicator(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        stderr_file = base_tr.output_path / "stderr.txt"
        stderr_content = """
        [NeMo Train] Trainer.fit` stopped: training completed successfully.
        """
        stderr_file.write_text(stderr_content)
        result = self.nemorun_tdef.was_run_successful(base_tr)
        assert not result.is_successful
        assert "'max_steps='" in result.error_message
        assert "'reached'" in result.error_message

    def test_missing_reached_indicator(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        stderr_file = base_tr.output_path / "stderr.txt"
        stderr_content = """
        [NeMo Train] Trainer.fit` stopped: `max_steps=100`.
        """
        stderr_file.write_text(stderr_content)
        result = self.nemorun_tdef.was_run_successful(base_tr)
        assert not result.is_successful
        assert "'reached'" in result.error_message
        assert "'max_steps='" not in result.error_message

    def test_missing_both_indicators(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        stderr_file = base_tr.output_path / "stderr.txt"
        stderr_content = """
        [NeMo Train] Trainer.fit stopped unexpectedly.
        """
        stderr_file.write_text(stderr_content)
        result = self.nemorun_tdef.was_run_successful(base_tr)
        assert not result.is_successful
        assert "'max_steps='" in result.error_message
        assert "'reached'" in result.error_message
