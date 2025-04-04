# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tarfile
from pathlib import Path


class CompressionUtility:
    """Utility class for compressing directories into tar-gz files."""

    @staticmethod
    def compress_directory(directory_path: Path, output_path: Path) -> None:
        """
        Compress the specified directory into a tar-gz file.

        Args:
            directory_path (Path): The path to the directory to compress.
            output_path (Path): The path where the tar-gz file will be saved.
        """
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(directory_path, arcname=directory_path.name)
