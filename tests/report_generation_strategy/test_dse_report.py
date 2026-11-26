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


import csv
from io import StringIO
from pathlib import Path

import pytest

import cloudai.configurator as configurator
from cloudai.report_generator.dse_report import load_trajectory_dataframe


@pytest.mark.parametrize(
    "rows",
    [
        [[1, {"x": "001"}, "010", []]],
        [[1, {"x": 1}, "", []]],
        [[1, {"x": float("nan")}, "nan", [float("nan")]]],
        [[1, {"x": 1}, "1", []], [2, {"x": 2}, "2.5", []]],
        [[1, {"model": {"layers": 8}, "empty": {}}, "2", [1.0]]],
        [],
    ],
    ids=["leading-zero", "empty-reward", "nan", "mixed-numbers", "nested-actions", "header-only"],
)
def test_report_reader_preserves_existing_frame_inference(tmp_path: Path, rows):
    import pandas as pd

    source = tmp_path / "trajectory.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "action", "reward", "observation"])
        for step, action, reward, observation in rows:
            writer.writerow([step, repr(action), reward, repr(observation)])
    original = source.read_bytes()
    expected = pd.read_csv(StringIO(original.decode()))
    loaded = load_trajectory_dataframe(tmp_path)
    assert loaded is not None
    path, frame = loaded
    assert path == source
    pd.testing.assert_frame_equal(frame, expected)
    assert source.read_bytes() == original


@pytest.mark.parametrize("artifact", ["absent", "orphan", "duplicate-header", "wide-row", "missing-metadata-step"])
def test_report_reader_uses_shared_integrity_boundary(tmp_path: Path, artifact: str):
    source = tmp_path / "trajectory.csv"
    if artifact == "orphan":
        (tmp_path / "metadata.csv").write_text("step,execution.ticket\n1,T-1\n")
    elif artifact == "duplicate-header":
        source.write_text("step,action,reward,reward,observation\n1,{},1,2,[]\n")
    elif artifact == "wide-row":
        source.write_text("step,action,reward,observation\n1,{},1,[],extra\n")
    elif artifact == "missing-metadata-step":
        source.write_text("step,action,reward,observation\n1,{},1,[]\n2,{},2,[]\n")
        (tmp_path / "metadata.csv").write_text("step,execution.ticket\n1,T-1\n")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    if artifact == "absent":
        assert load_trajectory_dataframe(tmp_path) is None
    else:
        with pytest.raises(configurator.TrajectoryLoadError):
            load_trajectory_dataframe(tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
