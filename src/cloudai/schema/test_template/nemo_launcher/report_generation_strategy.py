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
from typing import Optional

import pandas as pd

from cloudai import ReportGenerationStrategy
from cloudai.report_generator.tool.bokeh_report_tool import BokehReportTool
from cloudai.report_generator.tool.tensorboard_data_reader import TensorBoardDataReader


class NeMoLauncherReportGenerationStrategy(ReportGenerationStrategy):
    """
    Strategy for generating reports from NeMo launcher directories.

    Now updated to handle TensorBoard log files and visualize data using Bokeh plots.
    """

    def can_handle_directory(self, directory_path: str) -> bool:
        for _, __, files in os.walk(directory_path):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    return True
        return False

    def generate_report(self, test_name: str, directory_path: str, sol: Optional[float] = None) -> None:
        tags = ["train_step_timing in s"]
        data_reader = TensorBoardDataReader(directory_path)
        report_tool = BokehReportTool(directory_path)

        for tag in tags:
            data = data_reader.extract_data(tag)
            if data:
                df = pd.DataFrame(data, columns=["Step", tag])
                report_tool.add_linear_xy_line_plot(
                    title=f"{tag} over Time",
                    x_column="Step",
                    y_column=tag,
                    x_axis_label="Step",
                    df=df,
                    sol=sol,
                    color="black",
                )

        report_tool.finalize_report("cloudai_nemo_launcher_bokeh_report.html")
