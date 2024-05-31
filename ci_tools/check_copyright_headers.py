# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


def find_python_files(directory: str) -> List[str]:
    """
    Recursively find all Python files (.py) in the specified directory.

    Args:
        directory (str): The root directory to search for Python files.

    Returns:
        List[str]: A list of paths to Python files found within the directory.
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def check_copyright_header(file_path: str, header: str) -> Tuple[bool, str, str]:
    """
    Validate the presence of a copyright header in a Python file.

    Ensure correct formatting and adherence to PEP 8 regarding import statements.

    Args:
        file_path (str): Path to the file.
        header (str): Copyright header text.

    Returns:
        Tuple[bool, str, str]: Whether the header is correct, file path, and
        error message if applicable.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    header_lines = header.strip().split("\n")
    file_header_lines = [line.rstrip() for line in lines[: len(header_lines)]]

    if header_lines != file_header_lines:
        return False, file_path, "Header does not match"

    return True, file_path, ""


def main(directories: List[str], header: str):
    """
    Check Python files in specified directories for a correctly formatted copyright header.

    Args:
        directories (List[str]): Directories to check.
        header (str): Copyright header text.
    """
    incorrect_header_files = []

    for directory in directories:
        python_files = find_python_files(directory)
        for file_path in python_files:
            correct, path, error = check_copyright_header(file_path, header)
            if not correct:
                incorrect_header_files.append((path, error))

    if incorrect_header_files:
        print("Files with incorrect header formatting:")
        for file, error in incorrect_header_files:
            print(f"{file}: {error}")
        exit(1)
    else:
        print("All files have the correct header formatting.")


if __name__ == "__main__":
    dirs_to_check = ["src", "ci_tools"]
    copyright_header = """\
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
# limitations under the License."""
    main(dirs_to_check, copyright_header)
