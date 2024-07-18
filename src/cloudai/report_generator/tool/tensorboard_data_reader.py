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
from typing import List, Tuple


class TensorBoardDataReader:
    """
    Reads scalar data from TensorBoard log files for specified tags.

    Attributes
        directory_path (str): Path to the directory containing TensorBoard logs.
    """

    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    def extract_data(self, tag: str) -> List[Tuple[int, float]]:
        """
        Extract scalar data for a given tag from all TensorBoard log files found in the directory.

        Args:
            tag (str): The tag of the data to extract.

        Returns:
            List[Tuple[int, float]]: A list of (step, value) tuples.
        """
        from tbparse import SummaryReader  # lazy import to improve overall performance

        data = []
        for root, _, files in os.walk(self.directory_path):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    path = os.path.join(root, file)
                    reader = SummaryReader(path)
                    df = reader.scalars
                    if tag in df["tag"].values:
                        filtered_data = df[df["tag"] == tag]
                        data.extend(filtered_data[["step", "value"]].values)
        return data


def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python script_name.py <directory_path> <tag>")
        return

    directory_path = sys.argv[1]
    tag = sys.argv[2]

    try:
        reader = TensorBoardDataReader(directory_path)
        data = reader.extract_data(tag)
        if data:
            print(f"Data for tag '{tag}':")
            for step, value in data:
                print(f"Step: {step}, Value: {value}")
        else:
            print(f"No data found for tag '{tag}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
