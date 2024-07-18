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
from math import pi
from typing import List, Optional, Tuple

import pandas as pd
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJSTickFormatter, Range1d
from bokeh.plotting import figure, output_file, save

from cloudai.report_generator.util import bokeh_size_unit_js_tick_formatter, calculate_power_of_two_ticks


class BokehReportTool:
    """
    Tool for creating interactive Bokeh plots.

    Attributes
        output_directory (str): Directory to save the generated reports.
    """

    def __init__(self, output_directory: str):
        self.output_directory = output_directory
        self.plots = []

    def create_figure(
        self,
        title: str,
        x_axis_label: str,
        y_axis_label: str,
        y_range: Range1d,
        width: int = 500,
        height: int = 308,
        x_axis_type: str = "linear",
        tools: str = "pan,wheel_zoom,box_zoom,reset,save",
    ) -> figure:
        """
        Create a configured Bokeh figure with common settings.

        Args:
            title (str): Title of the plot.
            x_axis_label (str): Label for the x-axis.
            y_axis_label (str): Label for the y-axis.
            y_range (Range1d): Range for the y-axis, must be provided.
            width (int): Width of the plot.
            height (int): Height of the plot.
            x_axis_type (str): Type of the x-axis ('linear' or 'log').
            tools (str): Tools to include in the plot.

        Returns:
            figure: A Bokeh figure configured with the specified parameters.
        """
        plot = figure(
            title=title,
            width=width,
            height=height,
            x_axis_type=x_axis_type,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            tools=tools,
            y_range=y_range,
            align="center",
        )
        return plot

    def add_sol_line(
        self,
        plot: figure,
        df: pd.DataFrame,
        x_column: str,
        y_column: str,
        sol: Optional[float],
    ):
        """
        Add a Speed-of-Light (SOL) reference line to the given plot if an SOL value is provided.

        Args:
            plot (figure): The plot to which the SOL line will be added.
            df (pd.DataFrame): DataFrame containing the data for the plot.
            x_column (str): The data column used for x-axis values.
            y_column (str): The data column used for y-axis values.
            sol (Optional[float]): The SOL value to be plotted, if provided.
        """
        if sol is not None:  # Only add the SOL line if a value is provided
            sol_df = pd.DataFrame({x_column: df[x_column], y_column: [sol] * len(df)})
            sol_source = ColumnDataSource(sol_df)
            plot.line(
                x=x_column,
                y=y_column,
                source=sol_source,
                line_width=2,
                color="red",
                legend_label="SOL",
                line_dash="dashed",
            )

    def find_min_max(self, df: pd.DataFrame, column_name: str, sol: Optional[float] = None) -> Tuple[float, float]:
        """
        Find the minimum and maximum values of a specified column in a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            column_name (str): Name of the column to find the min and max values for.
            sol (Optional[float]): Optional value to compare against the maximum value.

        Returns:
            Tuple[float, float]: A tuple containing the minimum and maximum values.
        """
        min_val = df[column_name].min()
        max_val = df[column_name].max()
        if sol:
            max_val = max(max_val, sol)
        return min_val, max_val

    def add_linear_xy_line_plot(
        self,
        title: str,
        x_column: str,
        y_column: str,
        x_axis_label: str,
        df: pd.DataFrame,
        sol: Optional[float] = None,
        color: str = "black",
    ):
        """
        Add a line plot with linear axes to the report tool.

        Includes an optional reference line representing the speed of light (SOL) performance.

        Args:
            title (str): Title of the plot.
            x_column (str): The column name for the x-axis values.
            y_column (str): The column name for the y-axis values.
            x_axis_label (str): Label for the x-axis.
            df (pd.DataFrame): DataFrame containing the data.
            sol (Optional[float], optional): Value to plot as the SOL reference line.
            color (str, optional): Color of the line in the plot. Default is 'black'.
        """
        p = self.create_figure(
            title="CloudAI " + title,
            x_axis_label=x_axis_label,
            y_axis_label=y_column,
            y_range=Range1d(start=0, end=(max(df[y_column]) * 1.1)),
        )

        p.line(x=x_column, y=y_column, source=ColumnDataSource(df), line_width=2, color=color, legend_label=y_column)

        self.add_sol_line(p, df, x_column, y_column, sol)

        self.plots.append(p)

    def add_log_x_linear_y_single_line_plot(
        self,
        title: str,
        x_column: str,
        y_column: str,
        x_axis_label: str,
        y_axis_label: str,
        df: pd.DataFrame,
        sol: Optional[float] = None,
        color: str = "black",
    ):
        """
        Create a single line plot with a logarithmic x-axis and linear y-axis.

        Args:
            title (str): Title of the plot.
            x_column (str): The column used for the x-axis values.
            y_column (str): The column used for the y-axis values.
            x_axis_label (str): Label for the x-axis.
            y_axis_label (str): Label for the y-axis.
            df (pd.DataFrame): DataFrame containing the data.
            sol (Optional[float]): Speed-of-light performance reference line.
            color (str): Color of the line in the plot.

        This function sets up a Bokeh figure and plots a single line of data. It also
        optionally adds a reference line (SOL) if provided. The x-axis uses a logarithmic
        scale, and custom JavaScript is used for tick formatting to enhance readability.
        """
        x_min, x_max = self.find_min_max(df, x_column)
        y_min, y_max = self.find_min_max(df, y_column, sol)

        # Create a Bokeh figure with logarithmic x-axis
        p = self.create_figure(
            title="CloudAI " + title,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            x_axis_type="log",
            y_range=Range1d(start=0, end=y_max * 1.1),
        )

        # Add main line plot
        p.line(x=x_column, y=y_column, source=ColumnDataSource(df), line_width=2, color=color, legend_label=y_column)

        self.add_sol_line(p, df, x_column, y_column, sol)

        p.legend.location = "bottom_right"

        p.xaxis.ticker = calculate_power_of_two_ticks(x_min, x_max)
        p.xaxis.formatter = CustomJSTickFormatter(code=bokeh_size_unit_js_tick_formatter)
        p.xaxis.major_label_orientation = pi / 4

        # Append plot to internal list for future rendering
        self.plots.append(p)

    def add_log_x_linear_y_multi_line_plot(
        self,
        title: str,
        x_column: str,
        y_columns: List[Tuple[str, str]],
        x_axis_label: str,
        y_axis_label: str,
        df: pd.DataFrame,
        sol: Optional[float] = None,
    ):
        """
        Add a line plot with a logarithmic x-axis and linear y-axis for multiple datasets.

        Args:
            title (str): Title of the plot.
            x_column (str): The column used for the x-axis values.
            y_columns (List[Tuple[str, str]]): A list of tuples specifying the y-axis column names and their colors.
            x_axis_label (str): Label for the x-axis.
            y_axis_label (str): Label for the y-axis.
            df (pd.DataFrame): DataFrame containing the data.
            sol (Optional[float]): Speed-of-light performance reference line.
        """
        x_min, x_max = self.find_min_max(df, x_column)
        y_max = 0
        for y_column, _ in y_columns:
            _, col_max = self.find_min_max(df, y_column, sol)
            y_max = max(y_max, col_max)

        p = self.create_figure(
            title="CloudAI " + title,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            x_axis_type="log",
            y_range=Range1d(start=0, end=y_max * 1.1),
        )

        # Adding lines for each data type specified
        for y_column, color in y_columns:
            p.line(
                x=x_column, y=y_column, source=ColumnDataSource(df), line_width=2, color=color, legend_label=y_column
            )
            y_max = max(y_max, df[y_column].max())

        self.add_sol_line(p, df, x_column, "SOL", sol)

        p.legend.location = "bottom_right"

        # Setting up custom tick formatter for log scale readability
        p.xaxis.ticker = calculate_power_of_two_ticks(x_min, x_max)
        p.xaxis.formatter = CustomJSTickFormatter(code=bokeh_size_unit_js_tick_formatter)
        p.xaxis.major_label_orientation = pi / 4

        self.plots.append(p)

    def finalize_report(self, output_filename: str):
        """
        Save all accumulated plots to a single HTML file.

        Args:
            output_filename (str): output_filename to save the final report.
        """
        output_filepath = os.path.join(self.output_directory, output_filename)
        output_file(output_filepath)
        save(column(*self.plots))
        self.plots = []  # Clear the list after saving to prepare for future use.
