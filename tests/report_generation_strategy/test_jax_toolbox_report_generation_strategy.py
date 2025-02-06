# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import Mock

import pytest

from cloudai import Test, TestRun
from cloudai.schema.test_template.jax_toolbox.report_generation_strategy import (
    JaxToolboxReportGenerationStrategy,
)
from cloudai.test_definitions.gpt import GPTCmdArgs, GPTTestDefinition


@pytest.fixture
def jax_tr(tmp_path: Path) -> TestRun:
    test = Test(
        test_definition=GPTTestDefinition(
            name="nccl",
            description="desc",
            test_template_name="t",
            cmd_args=GPTCmdArgs(docker_image_url="docker://url", fdl_config="cfg"),
        ),
        test_template=Mock(),
    )
    return TestRun(name="nemo", test=test, num_nodes=1, nodes=[], output_path=tmp_path)


class TestJaxExtractTime:
    """Tests for the JaxToolboxReportGenerationStrategy class."""

    @pytest.fixture
    def js(self, jax_tr: TestRun) -> JaxToolboxReportGenerationStrategy:
        return JaxToolboxReportGenerationStrategy(jax_tr)

    def test_no_files(self, js: JaxToolboxReportGenerationStrategy) -> None:
        """Test that no times are extracted when no files are present."""
        assert js._extract_times() == []

    def test_no_matches(self, js: JaxToolboxReportGenerationStrategy) -> None:
        """Test that no times are extracted when no matching lines are present."""
        (js.test_run.output_path / "error-1.txt").write_text("fake line")
        assert js._extract_times() == []

    def test_one_match(self, js: JaxToolboxReportGenerationStrategy) -> None:
        """Test that the correct time is extracted when one matching line is present."""
        stdout_content = """
            "I0508 15:25:28.482553 140737334253888 programs.py:379] "
            "[PAX STATUS]: train_step() took 38.727223 seconds.\n"
        """
        with (js.test_run.output_path / "error-1.txt").open("w") as of:
            for _ in range(11):
                of.write(stdout_content)
        assert js._extract_times() == [38.727223]
