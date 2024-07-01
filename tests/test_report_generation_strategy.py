#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai.schema.test_template.jax_toolbox.report_generation_strategy import (
    JaxToolboxReportGenerationStrategy,
)


class TestJaxExtractTime:
    """Tests for the JaxToolboxReportGenerationStrategy class."""

    def setup_method(self) -> None:
        """Setup method for initializing JaxToolboxReportGenerationStrategy."""
        self.js = JaxToolboxReportGenerationStrategy()

    def test_no_files(self, tmp_path: Path) -> None:
        """Test that no times are extracted when no files are present."""
        assert self.js._extract_times(str(tmp_path)) == []

    def test_no_matches(self, tmp_path: Path) -> None:
        """Test that no times are extracted when no matching lines are present."""
        (tmp_path / "error-1.txt").write_text("fake line")
        assert self.js._extract_times(str(tmp_path)) == []

    def test_one_match(self, tmp_path: Path) -> None:
        """Test that the correct time is extracted when one matching line is present."""
        err_file = tmp_path / "error-1.txt"
        sample_line = (
            "I0508 15:25:28.482553 140737334253888 programs.py:379] "
            "[PAX STATUS]: train_step() took 38.727223 seconds.\n"
        )
        with err_file.open("w") as f:
            for _ in range(11):
                f.write(sample_line)
        assert self.js._extract_times(str(err_file.parent)) == [38.727223]
