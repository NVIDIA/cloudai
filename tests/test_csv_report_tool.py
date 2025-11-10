# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pandas as pd
import pytest

from cloudai.report_generator.tool.csv_report_tool import CSVReportTool


@pytest.fixture
def output_directory(tmpdir) -> Path:
    return Path(tmpdir.mkdir("reports"))


@pytest.fixture
def example_dataframe() -> pd.DataFrame:
    data = {"A": [1, 2, 3], "B": [4, 5, 6]}
    return pd.DataFrame(data)


def test_set_dataframe(example_dataframe: pd.DataFrame):
    csv_tool = CSVReportTool(output_directory=Path("."))
    csv_tool.set_dataframe(example_dataframe)
    assert csv_tool.dataframe is not None, "The dataframe was not set."
    assert csv_tool.dataframe.equals(example_dataframe), "The dataframe was not set correctly."


def test_finalize_report_no_dataframe(output_directory: Path):
    csv_tool = CSVReportTool(output_directory=output_directory)
    with pytest.raises(ValueError, match=r"No DataFrame has been set for the report."):
        csv_tool.finalize_report(Path("report.csv"))


def test_finalize_report(output_directory: Path, example_dataframe: pd.DataFrame):
    csv_tool = CSVReportTool(output_directory=output_directory)
    csv_tool.set_dataframe(example_dataframe)
    output_filename = Path("report.csv")
    csv_tool.finalize_report(output_filename)

    output_filepath = output_directory / output_filename
    assert output_filepath.exists(), "The report file was not created."

    saved_df = pd.read_csv(output_filepath)
    assert saved_df.equals(example_dataframe), "The saved dataframe does not match the original dataframe."


def test_finalize_report_resets_dataframe(output_directory: Path, example_dataframe: pd.DataFrame):
    csv_tool = CSVReportTool(output_directory=output_directory)
    csv_tool.set_dataframe(example_dataframe)
    csv_tool.finalize_report(Path("report.csv"))
    assert csv_tool.dataframe is None, "The dataframe was not reset after finalizing the report."
