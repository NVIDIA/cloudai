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

import pytest

from cloudai.util.compression_utility import CompressionUtility


@pytest.fixture
def sample_directory(tmp_path: Path) -> Path:
    d = tmp_path / "sample"
    d.mkdir()
    (d / "file1.txt").write_text("content1")
    sub = d / "subfolder"
    sub.mkdir()
    (sub / "file2.txt").write_text("content2")
    return d


@pytest.fixture
def output_path(tmp_path: Path, sample_directory: Path) -> Path:
    return tmp_path / f"{sample_directory.name}.tar.gz"


def test_compress_directory(sample_directory: Path, output_path: Path) -> None:
    CompressionUtility.compress_directory(sample_directory, output_path)
    assert output_path.exists()
    with tarfile.open(output_path, "r:gz") as tar:
        names = tar.getnames()
    assert sample_directory.name in names


def test_compressed_content(sample_directory: Path, output_path: Path, tmp_path: Path) -> None:
    CompressionUtility.compress_directory(sample_directory, output_path)
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    with tarfile.open(output_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    sample_extracted = extract_dir / sample_directory.name
    assert (sample_extracted / "file1.txt").read_text() == "content1"
    assert (sample_extracted / "subfolder" / "file2.txt").read_text() == "content2"
