# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path

import pytest

from cloudai.workloads.osu_bench.report_generation_strategy import extract_osu_bench_data

OSU_MULTIPLE_BW = """\
# OSU MPI Multiple Bandwidth / Message Rate Test v7.4
# [ pairs: 1 ] [ window size: 64 ]
# Datatype: MPI_CHAR.
# Size                  MB/s        Messages/s
1                       2.36        2356892.44
4194304             26903.29           6414.24
"""


OSU_ALLGATHER_LAT = """\
# OSU MPI Allgather Latency Test v7.4
# Datatype: MPI_CHAR.
# Size       Avg Latency(us)   Min Latency(us)   Max Latency(us)  Iterations
1                       2.81              1.89              3.73          10
1048576               104.30            104.26            104.33          10
"""


OSU_BW = """\
# OSU MPI Bandwidth Test v7.4
# Datatype: MPI_CHAR.
# Size      Bandwidth (MB/s)
1                       2.32
4194304             27161.33
"""

OSU_MULTI_LAT = """\
# OSU MPI Multi Latency Test v7.4
# Datatype: MPI_CHAR.
# Size       Avg Latency(us)
1                       1.88
2                       1.84
4                       1.88
8                       1.91
16                      1.87
32                      2.01
"""


def test_osu_multiple_bandwidth_message_rate_parsing(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.txt"
    stdout.write_text(OSU_MULTIPLE_BW)

    df = extract_osu_bench_data(stdout)
    assert list(df.columns) == ["size", "mb_sec", "messages_sec"]
    assert df.shape == (2, 3)
    assert df["size"].iloc[0] == 1
    assert df["mb_sec"].iloc[0] == pytest.approx(2.36)
    assert df["messages_sec"].iloc[0] == pytest.approx(2356892.44)
    assert df["size"].iloc[-1] == 4194304
    assert df["mb_sec"].iloc[-1] == pytest.approx(26903.29)
    assert df["messages_sec"].iloc[-1] == pytest.approx(6414.24)


def test_osu_latency_parsing(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.txt"
    stdout.write_text(OSU_ALLGATHER_LAT)

    df = extract_osu_bench_data(stdout)
    assert list(df.columns) == ["size", "avg_lat"]
    assert df.shape == (2, 2)
    assert df["size"].iloc[0] == 1
    assert df["avg_lat"].iloc[0] == pytest.approx(2.81)
    assert df["size"].iloc[-1] == 1048576
    assert df["avg_lat"].iloc[-1] == pytest.approx(104.30)


def test_osu_bandwidth_parsing(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.txt"
    stdout.write_text(OSU_BW)

    df = extract_osu_bench_data(stdout)
    assert list(df.columns) == ["size", "mb_sec"]
    assert df.shape == (2, 2)
    assert df["size"].iloc[0] == 1
    assert df["mb_sec"].iloc[0] == pytest.approx(2.32)
    assert df["size"].iloc[-1] == 4194304
    assert df["mb_sec"].iloc[-1] == pytest.approx(27161.33)


def test_osu_multi_latency_short_header_parsing(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.txt"
    stdout.write_text(OSU_MULTI_LAT)

    df = extract_osu_bench_data(stdout)
    assert list(df.columns) == ["size", "avg_lat"]
    assert df.shape == (6, 2)
    assert df["size"].tolist() == [1, 2, 4, 8, 16, 32]
    assert df["avg_lat"].tolist() == pytest.approx([1.88, 1.84, 1.88, 1.91, 1.87, 2.01])


def test_extract_osu_bench_data_file_not_found_returns_empty_dataframe(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.txt"
    df = extract_osu_bench_data(missing)
    assert df.empty


def test_extract_osu_bench_data_empty_file_returns_empty_dataframe(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.txt"
    stdout.write_text("")
    df = extract_osu_bench_data(stdout)
    assert df.empty


def test_extract_osu_bench_data_no_recognizable_header_returns_empty_dataframe(tmp_path: Path) -> None:
    # e.g. osu_hello or other benchmark with no OSU latency/bandwidth header
    stdout = tmp_path / "stdout.txt"
    stdout.write_text("Hello world from rank 0\nHello world from rank 1\n")
    df = extract_osu_bench_data(stdout)
    assert df.empty
