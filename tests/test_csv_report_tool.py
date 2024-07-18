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

import os

import pandas as pd
import pytest
from cloudai.report_generator.tool.csv_report_tool import CSVReportTool


@pytest.fixture
def output_directory(tmpdir):
    return tmpdir.mkdir("reports")


@pytest.fixture
def example_dataframe():
    data = {"A": [1, 2, 3], "B": [4, 5, 6]}
    return pd.DataFrame(data)


def test_set_dataframe(example_dataframe):
    csv_tool = CSVReportTool(output_directory=".")
    csv_tool.set_dataframe(example_dataframe)
    assert csv_tool.dataframe is not None, "The dataframe was not set."
    assert csv_tool.dataframe.equals(example_dataframe), "The dataframe was not set correctly."


def test_finalize_report_no_dataframe(output_directory):
    csv_tool = CSVReportTool(output_directory=str(output_directory))
    with pytest.raises(ValueError, match="No DataFrame has been set for the report."):
        csv_tool.finalize_report("report.csv")


def test_finalize_report(output_directory, example_dataframe):
    csv_tool = CSVReportTool(output_directory=str(output_directory))
    csv_tool.set_dataframe(example_dataframe)
    output_filename = "report.csv"
    csv_tool.finalize_report(output_filename)

    output_filepath = os.path.join(str(output_directory), output_filename)
    assert os.path.exists(output_filepath), "The report file was not created."

    saved_df = pd.read_csv(output_filepath)
    assert saved_df.equals(example_dataframe), "The saved dataframe does not match the original dataframe."


def test_finalize_report_resets_dataframe(output_directory, example_dataframe):
    csv_tool = CSVReportTool(output_directory=str(output_directory))
    csv_tool.set_dataframe(example_dataframe)
    csv_tool.finalize_report("report.csv")
    assert csv_tool.dataframe is None, "The dataframe was not reset after finalizing the report."
