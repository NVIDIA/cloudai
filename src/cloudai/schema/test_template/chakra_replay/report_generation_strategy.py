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

import os
import re
from math import pi
from typing import Dict, Optional

import pandas as pd
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, DataTable, Div, TableColumn, Title
from bokeh.palettes import Turbo256
from bokeh.plotting import figure, output_file, save
from bokeh.transform import cumsum

from cloudai import ReportGenerationStrategy


class ChakraReplayReportGenerationStrategy(ReportGenerationStrategy):
    """
    Strategy for generating reports from Chakra replay directories using Bokeh for graphical summaries of the data.

    This class provides methods to check if a directory can be handled, extract communication data, latency tables, and
    tensor sizes from stdout files, and generate a report using Bokeh.
    """

    def can_handle_directory(self, directory_path: str) -> bool:
        stdout_path = os.path.join(directory_path, "stdout.txt")
        if os.path.exists(stdout_path):
            with open(stdout_path, "r") as file:
                if re.search(r"Hello from Rank \d+: \[Rank\s+\d+\]", file.read()):
                    return True
        return False

    def generate_report(self, test_name: str, directory_path: str, sol: Optional[float] = None) -> None:
        stdout_path = os.path.join(directory_path, "stdout.txt")
        if not os.path.isfile(stdout_path):
            return

        comms_data = self._extract_comms_data(stdout_path)
        latency_tables = self._extract_latency_tables(stdout_path)
        tensor_sizes = self._extract_tensor_sizes(stdout_path)

        self._generate_bokeh_content(comms_data, latency_tables, tensor_sizes, directory_path)

    def _extract_comms_data(self, file_path: str) -> Dict[str, int]:
        """
        Extract the number of times each communication operation is called from the specified file.

        Args:
            file_path: Path to the file containing stdout data.

        Returns:
            A dictionary with operation names as keys and their call counts as
            values.
        """
        comms_data = {}
        pattern = r"Replayed (\d+) (\w+)"

        with open(file_path, "r") as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    count, operation = match.groups()
                    comms_data[operation] = int(count)

        return comms_data

    def _extract_latency_tables(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Extract latency distribution tables for communication operations from the specified file.

        Args:
            file_path: Path to the file containing stdout data.

        Returns:
            A dictionary with the operation names as keys and the corresponding
            latency distribution DataFrames as values.
        """
        comms_data = {}
        headers = ["Total", "Max", "Min", "Average", "p50", "p95"]

        with open(file_path, "r") as file:
            lines = file.readlines()

        op_name = None
        data_started = False
        data_values = []

        for line in lines:
            if "Replayed" in line and "Latency (us)" not in line:
                op_name_match = re.search(r"Replayed \d+ (\w+)", line)
                if op_name_match:
                    op_name = op_name_match.group(1)
            elif "Latency (us)" in line:
                data_started = True
            elif data_started:
                current_values = [float(val) for val in re.findall(r"\b\d+\.?\d*\b", line)]
                if current_values:
                    data_values.append(current_values)
                else:
                    if op_name and data_values:
                        df = pd.DataFrame(data_values, columns=headers[: len(data_values[0])])
                        comms_data[op_name] = df
                        data_started = False
                        data_values = []

        if op_name and data_values:
            df = pd.DataFrame(data_values, columns=headers[: len(data_values[0])])
            comms_data[op_name] = df

        return comms_data

    def _extract_tensor_sizes(self, file_path: str) -> Dict[str, Dict[str, pd.DataFrame]]:  # noqa: C901
        """
        Extract input and output tensor size distribution tables from the specified file.

        Args:
            file_path: Path to the file containing stdout data.

        Returns:
            A dictionary with operation names as keys and another dictionary as
            values, which contains two DataFrames for input and output tensor
            sizes respectively.
        """
        tensor_sizes = {}
        headers = ["Total (MB)", "Max.", "Min.", "Average", "p50", "p95"]

        with open(file_path, "r") as file:
            lines = file.readlines()

        op_name = None
        section = None  # To track if we are in 'Input' or 'Output' section
        data_values = []

        for line in lines:
            if "+++++" in line:
                # Skip lines that are not related to tensor sizes
                continue

            if "Performance of replayed comms" in line:
                # Stop parsing as we've reached performance summary
                break

            # Check for operation name line. The operation name appears after '----' line
            if "----" in line:
                continue  # Skip the '----' line itself

            op_name_match = re.search(r"\+ (\d+) (\w+)", line)
            if op_name_match:
                if op_name and section and data_values:
                    df = pd.DataFrame(data_values, columns=headers)
                    tensor_sizes[op_name][section] = df

                # Reset for new operation
                op_name = op_name_match.group(2)
                tensor_sizes[op_name] = {"Input": None, "Output": None}
                data_values = []  # Clear previous data
                section = None  # Reset section for new operation

            elif "Input tensors" in line or "Output tensors" in line:
                # Save previous section (Input/Output) before starting new section
                if op_name and section and data_values:
                    df = pd.DataFrame(data_values, columns=headers)
                    tensor_sizes[op_name][section] = df
                    data_values = []  # Reset for new section

                section = "Input" if "Input" in line else "Output"

            elif op_name and section:
                current_values = [float(val) for val in re.findall(r"\b\d+\.?\d*\b", line)]
                if current_values:
                    data_values.append(current_values)

        # Save the last collected data for the last operation
        if op_name and section and data_values:
            df = pd.DataFrame(data_values, columns=headers)
            tensor_sizes[op_name][section] = df

        return tensor_sizes

    def _generate_bokeh_content(
        self,
        comms_data: Dict[str, int],
        latency_tables: Dict[str, pd.DataFrame],
        tensor_sizes: Dict[str, Dict[str, pd.DataFrame]],
        directory_path: str,
    ) -> None:
        """
        Generate Bokeh visualizations for the report.

        Including pie charts for communication operations distribution and DataTables for latency metrics and tensor
        sizes.

        Args:
            comms_data: A dictionary with operation names as keys and their call
                        counts as values.
            latency_tables: A dictionary with operation names as keys and the
                            corresponding latency distribution DataFrames as values.
            tensor_sizes: A dictionary with operation names as keys and another
                          dictionary as values, containing DataFrames for input
                          and output tensor sizes.
            directory_path: The directory path to save the report.
        """
        # Generate and configure pie chart for communications data
        data = pd.Series(comms_data).reset_index(name="value").rename(columns={"index": "comm"})
        data["angle"] = data["value"] / data["value"].sum() * 2 * pi
        data["color"] = [Turbo256[i * int(256 / len(data))] for i in range(len(data))]
        data["percentage"] = (data["value"] / data["value"].sum() * 100).apply(lambda x: f"{x:.2f}%")

        source = ColumnDataSource(data)
        p = figure(
            height=350,
            toolbar_location="right",
            tools="hover",
            tooltips="@comm: @value (@percentage)",
            x_range=(-0.5, 1.5),
        )
        p.add_layout(Title(text="Comm Types Distribution", align="center"), "above")
        p.wedge(
            x=0,
            y=1,
            radius=0.4,
            start_angle=cumsum("angle", include_zero=True),
            end_angle=cumsum("angle"),
            line_color="white",
            fill_color="color",
            legend_field="comm",
            source=source,
        )

        # Generate DataTable for latency metrics
        merged_df = pd.DataFrame(columns=["Comm Type"])
        for comm_type, df in latency_tables.items():
            df["Comm Type"] = comm_type
            df = df[["Comm Type"] + [col for col in df.columns if col != "Comm Type"]]
            merged_df = pd.concat([merged_df, df], ignore_index=True)

        table_title_div = Div(
            text="<h2>Latency Metrics by Communication Type</h2>",
            sizing_mode="stretch_width",
        )
        source = ColumnDataSource(merged_df)
        columns = [TableColumn(field=col, title=col) for col in merged_df.columns]
        data_table = DataTable(
            source=source,
            columns=columns,
            width=800,
            height=280,
            index_position=None,
        )

        layout = column(p, table_title_div, data_table)

        # Generate merged DataFrame for tensor sizes
        merged_tensor_sizes = self._transform_and_merge_tensor_sizes(tensor_sizes)

        # Add a title for the merged table
        table_title_div = Div(
            text="<h2>Tensor Sizes by Communication Type</h2>",
            sizing_mode="stretch_width",
        )

        # Create DataTable from the merged DataFrame
        source = ColumnDataSource(merged_tensor_sizes)
        columns = [TableColumn(field=col, title=col) for col in merged_tensor_sizes.columns]
        data_table = DataTable(
            source=source,
            columns=columns,
            width=800,
            height=280,
            index_position=None,
        )

        # Layout configuration (assuming latency data table layout is already defined as `layout_latency`)
        layout_final = column(layout, table_title_div, data_table)  # Combine layouts

        # Generate and save the report
        output_filepath = os.path.join(directory_path, "chakra_replay_report.html")
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        output_file(output_filepath)
        save(layout_final)  # Generates an HTML file with the specified filename

    def _transform_and_merge_tensor_sizes(self, tensor_sizes):
        """
        Transform and merges tensor size data from input and output into a single DataFrame.

        Args:
            tensor_sizes: A dictionary with operation names as keys and dictionaries
            as values, where each inner dictionary contains DataFrames for input
            and output tensor sizes.

        Returns:
            A single merged DataFrame with 'Comm Type' and 'Type' columns added.
        """
        merged_df = pd.DataFrame()

        for op_name, sizes in tensor_sizes.items():
            for io_type, df in sizes.items():
                if df is not None:
                    df["Comm Type"] = op_name
                    df["Type"] = io_type.capitalize()  # 'Input' or 'Output'
                    merged_df = pd.concat([merged_df, df], ignore_index=True)

        # Reorder columns to have 'Comm Type' and 'Type' at the beginning
        cols = ["Comm Type", "Type"] + [col for col in merged_df.columns if col not in ["Comm Type", "Type"]]
        merged_df = merged_df[cols]

        return merged_df
