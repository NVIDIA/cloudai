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

from typing import List, Tuple

import numpy as np
import pandas as pd

bokeh_size_unit_js_tick_formatter = """
    function tick_formatter(tick) {
        var units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB'];
        var i = 0;
        while (tick >= 1024 && i < units.length - 1) {
            tick /= 1024;
            i++;
        }
        return tick.toFixed(1) + units[i];
    }
    return tick_formatter(tick);
    """


def calculate_power_of_two_ticks(min_val: float, max_val: float) -> List[float]:
    """
    Calculate tick locations that are powers of 2 within the specified range.

    Args:
        min_val (float): Minimum value of the data range in bytes.
        max_val (float): Maximum value of the data range in bytes.

    Returns:
        List[float]: A list of tick locations that are powers of 2.
    """
    min_val = float(max(min_val, 1.0))
    min_exp = np.floor(np.log2(min_val))
    max_exp = np.ceil(np.log2(max_val))
    return [2**exp for exp in range(int(min_exp), int(max_exp) + 1)]


def bytes_to_human_readable(num_bytes: float) -> str:
    """
    Convert a number of bytes into a human-readable string with units.

    Args:
        num_bytes (float): The number of bytes.

    Returns:
        str: A human-readable string with units.
    """
    for unit in ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}YB"


def add_human_readable_sizes(
    df: pd.DataFrame,
    input_column: str,
    output_column: str,
) -> pd.DataFrame:
    """
    Add a human-readable size column to a DataFrame based on specified input column containing numerical data.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        input_column (str): The name of the column with original numerical values.
        output_column (str): The name of the column to store human-readable values.

    Returns:
        pd.DataFrame: Updated DataFrame with an additional human-readable column.
    """
    human_readable_sizes = [bytes_to_human_readable(int(size)) for size in df[input_column]]
    df[output_column] = human_readable_sizes
    return df


def generate_power_of_two_ticks(min_val: float, max_val: float) -> List[int]:
    """
    Generate tick locations that are powers of 2 within the specified range.

    Args:
        min_val (float): Minimum value of the data range.
        max_val (float): Maximum value of the data range.

    Returns:
        List[int]: A list of tick locations that are powers of 2.
    """
    min_exp = np.floor(np.log2(min_val))
    max_exp = np.ceil(np.log2(max_val))
    return [2**exp for exp in range(int(min_exp), int(max_exp) + 1)]


def adjust_scale(df: pd.DataFrame, input_column: str, output_column: str) -> Tuple[pd.DataFrame, str]:
    """
    Adjust the numerical scale of values in a DataFrame column from bytes to the most appropriate unit.

    Based on the maximum value in the dataset, and stores the adjusted values in a new column.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        input_column (str): The name of the column with original values.
        output_column (str): The name of the column to store scaled values.

    Returns:
        Tuple[pd.DataFrame, str]: The modified DataFrame and the unit of measurement used.
    """
    size = df[input_column].astype(int)
    max_size = size.max()
    unit = "B"  # Default unit is Bytes
    factor = 1  # Default factor for Bytes

    if max_size >= 1024**3:  # Greater than or equal to 1 GB
        unit = "GB"
        factor = 1024**3
    elif max_size >= 1024**2:  # Greater than or equal to 1 MB
        unit = "MB"
        factor = 1024**2
    elif max_size >= 1024:  # Greater than or equal to 1 KB
        unit = "KB"
        factor = 1024

    df[output_column] = size / factor
    return df, unit
