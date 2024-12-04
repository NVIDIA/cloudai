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

from math import pi
from pathlib import Path
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
        output_directory (Path): Directory to save the generated reports.
    """

    def __init__(self, output_directory: Path):
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
        x_range: Optional[Range1d] = None,
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
            x_range (Range1d): Range for the x-axis, optional.

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

        if x_range is not None:
            plot.x_range = x_range

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

    def add_single_point_plot(
        self,
        df: pd.DataFrame,
        title: str,
        x_column: str,
        y_columns: List[Tuple[str, str]],
        x_axis_label: str,
        y_axis_label: str,
    ):
        """
        Create a scatter plot for a single data point.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            title (str): Title of the plot.
            x_column (str): The column used for the x-axis value.
            y_columns (List[Tuple[str, str]]): A list of tuples specifying the y-axis column names and their colors.
            x_axis_label (str): Label for the x-axis.
            y_axis_label (str): Label for the y-axis.

        Returns:
            figure: A Bokeh figure with scatter plots.
        """
        single_x = df[x_column].iloc[0]
        x_padding = single_x * 0.1 if single_x > 0 else 1
        y_values = [df[y_column].iloc[0] for y_column, _ in y_columns]
        y_max = max(y_values)
        y_padding = y_max * 0.1 if y_max > 0 else 1

        p = self.create_figure(
            title="CloudAI " + title,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            x_axis_type="linear",
            y_range=Range1d(start=0, end=y_max + y_padding),
            x_range=Range1d(start=single_x - x_padding, end=single_x + x_padding),
        )

        for (y_column, color), single_y in zip(y_columns, y_values):
            p.scatter(x=[single_x], y=[single_y], size=6, color=color, legend_label=y_column)

        return p

    def add_multiple_messages_multi_lines_plot(
        self,
        df: pd.DataFrame,
        title: str,
        x_column: str,
        y_columns: List[Tuple[str, str]],
        x_axis_label: str,
        y_axis_label: str,
    ):
        """
        Create lines plot for multiple message sizes.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            title (str): Title of the plot.
            x_column (str): The column used for the x-axis values.
            y_columns (List[Tuple[str, str]]): A list of tuples specifying the y-axis column names and their colors.
            x_axis_label (str): Label for the x-axis.
            y_axis_label (str): Label for the y-axis.

        Returns:
            figure: A Bokeh figure with line plots for multiple y-values.
        """
        if len(y_columns) == 1:
            grouped = df.groupby(x_column)[y_columns[0][0]].agg(["min", "max", "mean"]).reset_index()
            grouped.columns = [x_column, "min", "max", "avg"]
            y_max = grouped[["min", "max", "avg"]].values.max()

            p = self.create_figure(
                title="CloudAI " + title,
                x_axis_label=x_axis_label,
                y_axis_label=y_axis_label,
                x_axis_type="log",
                y_range=Range1d(start=0, end=y_max * 1.1),
            )

            # Plot min, max, and avg lines
            p.line(
                x=x_column,
                y="min",
                source=ColumnDataSource(grouped),
                color="blue",
                legend_label=f"{y_columns[0][0]} Min",
            )

            p.line(
                x=x_column,
                y="max",
                source=ColumnDataSource(grouped),
                color="red",
                legend_label=f"{y_columns[0][0]} Max",
            )

            p.line(
                x=x_column,
                y="avg",
                source=ColumnDataSource(grouped),
                color="green",
                legend_label=f"{y_columns[0][0]} Avg",
            )

        else:
            grouped = df.groupby(x_column)[[y_column for y_column, _ in y_columns]].agg(["mean"]).reset_index()
            grouped.columns = [x_column] + [f"{y_column}_avg" for y_column, _ in y_columns]
            y_max = grouped.iloc[:, 1:].max().max()

            p = self.create_figure(
                title="CloudAI " + title,
                x_axis_label=x_axis_label,
                y_axis_label=y_axis_label,
                x_axis_type="log",
                y_range=Range1d(start=0, end=y_max * 1.1),
            )

            # Plot avg line for each y_column
            for y_column, color in y_columns:
                p.line(
                    x=x_column,
                    y=f"{y_column}_avg",
                    source=ColumnDataSource(grouped),
                    line_width=2,
                    color=color,
                    legend_label=f"{y_column} Avg",
                )

        return p

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
        if len(df) == 1:
            p = self.add_single_point_plot(df, title, x_column, y_columns, x_axis_label, y_axis_label)

        else:
            if df[x_column].duplicated().any():
                p = self.add_multiple_messages_multi_lines_plot(
                    df, title, x_column, y_columns, x_axis_label, y_axis_label
                )

            else:
                x_min, x_max = self.find_min_max(df, x_column)
                y_max = 0
                for y_column, _ in y_columns:
                    _, col_max = self.find_min_max(df, y_column, sol)
                    y_max = max(y_max, col_max)

                x_axis_type = "log"
                x_range = None

                # Check if x_min equals x_max - constant message size
                if x_min == x_max:
                    # Use iteration number as x-axis
                    df["iteration"] = range(1, len(df) + 1)
                    x_column = "iteration"
                    x_axis_label = "Iteration"
                    x_axis_type = "linear"
                    x_range = Range1d(start=1, end=len(df))

                p = self.create_figure(
                    title="CloudAI " + title,
                    x_axis_label=x_axis_label,
                    y_axis_label=y_axis_label,
                    x_axis_type=x_axis_type,
                    y_range=Range1d(start=0, end=y_max * 1.1),
                    x_range=x_range,
                )

                # Adding lines for each data type specified
                for y_column, color in y_columns:
                    p.line(
                        x=x_column,
                        y=y_column,
                        source=ColumnDataSource(df),
                        line_width=2,
                        color=color,
                        legend_label=y_column,
                    )

                    y_max = max(y_max, df[y_column].max())

                self.add_sol_line(p, df, x_column, "SOL", sol)

                if x_axis_type == "log":
                    # Setting up custom tick formatter for log scale readability
                    p.xaxis.ticker = calculate_power_of_two_ticks(x_min, x_max)
                    p.xaxis.formatter = CustomJSTickFormatter(code=bokeh_size_unit_js_tick_formatter)
                    p.xaxis.major_label_orientation = pi / 4

        p.legend.location = "bottom_right"
        self.plots.append(p)

    def finalize_report(self, output_filename: Path):
        """
        Save all accumulated plots to a single HTML file.

        Args:
            output_filename (Path): Path to save the final report.
        """
        output_filepath = self.output_directory / output_filename
        output_file(output_filepath)
        save(column(*self.plots))
        self.plots = []  # Clear the list after saving to prepare for future use.
