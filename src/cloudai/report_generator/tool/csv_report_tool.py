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

from .report_tool_interface import ReportToolInterface


class CSVReportTool(ReportToolInterface):
    """
    Tool for creating CSV reports.

    Attributes
        output_directory (str): Directory to save the generated reports.
        dataframe (pd.DataFrame): DataFrame containing the data to be saved.
    """

    def __init__(self, output_directory: str):
        self.output_directory = output_directory
        self.dataframe = None

    def set_dataframe(self, df: pd.DataFrame):
        """
        Set the DataFrame to be used in the report.

        Args:
            df (pd.DataFrame): DataFrame containing the data to be saved.
        """
        self.dataframe = df

    def finalize_report(self, output_filename: str):
        """
        Save the DataFrame to a CSV file.

        Args:
            output_filename (str): The filename to save the final report.
        """
        if self.dataframe is None:
            raise ValueError("No DataFrame has been set for the report.")

        output_filepath = os.path.join(self.output_directory, output_filename)
        self.dataframe.to_csv(output_filepath, index=False)
        self.dataframe = None
