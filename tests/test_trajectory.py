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

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import stat
from ast import literal_eval
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import pandas as pd
import pytest

import cloudai.configurator as configurator
from cloudai.configurator import Trajectory


def test_append_flattens_domains_into_dataframe_columns(tmp_path: Path) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)

    row = trajectory.append(
        step=1,
        action={"model": {"layers": 8}},
        reward=0.95,
        observation={"throughput": 120.0},
        env_params={"network_speed": 100},
        logging={"gpu_power_watts": 610.0},
    )

    expected = {
        "step": 1,
        "action.model.layers": 8,
        "reward": 0.95,
        "observation.throughput": 120.0,
        "env_params.network_speed": 100,
        "logging.gpu_power_watts": 610.0,
    }
    assert row.to_dict() == expected
    assert trajectory.dataframe.to_dict(orient="records") == [expected]
    assert all(pd.api.types.is_object_dtype(dtype) for dtype in trajectory.dataframe.dtypes)

    core = pd.read_csv(trajectory.output_path)
    assert list(core.columns) == ["step", "action", "reward", "observation"]
    action = core.loc[0, "action"]
    observation = core.loc[0, "observation"]
    assert isinstance(action, str)
    assert isinstance(observation, str)
    assert literal_eval(action) == {"model": {"layers": 8}}
    assert core.loc[0, "reward"] == 0.95
    assert literal_eval(observation) == [120.0]

    metadata = pd.read_csv(trajectory.metadata_path)
    assert metadata.to_dict(orient="records") == [
        {
            "step": 1,
            "env_params.network_speed": 100,
            "logging.gpu_power_watts": 610.0,
        }
    ]


def test_dataframe_and_returned_rows_are_copies(tmp_path: Path) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)
    samples = [1, 2]
    row = trajectory.append(
        step=1,
        action={"x": 1},
        reward=1.0,
        observation={"metric": 2.0},
        logging={"samples": samples},
    )

    samples.append(3)
    row["reward"] = 99.0
    row_samples = row["logging.samples"]
    assert isinstance(row_samples, list)
    row_samples.append(4)
    dataframe = trajectory.dataframe
    dataframe.loc[0, "reward"] = 88.0
    dataframe_samples = dataframe.loc[0, "logging.samples"]
    assert isinstance(dataframe_samples, list)
    dataframe_samples.append(5)
    found = trajectory.find(action={"x": 1})
    assert found is not None
    found_samples = found["logging.samples"]
    assert isinstance(found_samples, list)
    found_samples.append(6)

    assert trajectory.dataframe.loc[0, "reward"] == 1.0
    assert trajectory.dataframe.loc[0, "logging.samples"] == [1, 2]


def test_find_matches_flattened_subset_and_preserves_exact_types(tmp_path: Path) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)
    first = trajectory.append(
        step=1, action={"x": 1.0}, reward=1.0, observation={"metric": 1.0}, env_params={"speed": 2}
    )
    trajectory.append(step=2, action={"x": 1}, reward=2.0, observation={"metric": 2.0}, env_params={"speed": 2})

    match = trajectory.find(action={"x": 1.0}, env_params={"speed": 2})

    assert match is not None
    assert match.to_dict() == first.to_dict()
    assert trajectory.find(action={"x": True}, env_params={"speed": 2}) is None


def test_find_returns_none_for_unknown_columns(tmp_path: Path) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)
    trajectory.append(step=1, action={"x": 1}, reward=1.0, observation={"metric": 1.0})

    assert trajectory.find(env_params={"speed": 2}) is None


def test_find_separates_values_that_share_an_index_key(tmp_path: Path) -> None:
    """A nested mapping key of 1 and of "1" are different values and must not collide."""
    trajectory = Trajectory(iteration_dir=tmp_path)
    integer_key = trajectory.append(step=1, action=[{1: "a"}], reward=1.0, observation={"metric": 1.0})
    string_key = trajectory.append(step=2, action=[{"1": "a"}], reward=2.0, observation={"metric": 2.0})

    integer_match = trajectory.find(action=[{1: "a"}])
    string_match = trajectory.find(action=[{"1": "a"}])

    assert integer_match is not None and string_match is not None
    assert integer_match["step"] == integer_key["step"]
    assert string_match["step"] == string_key["step"]


@pytest.mark.parametrize(
    "appended, queried",
    [
        pytest.param([{1: "a"}], [{True: "a"}], id="mapping-keys-that-compare-equal"),
        pytest.param([{1}], [{1.0}], id="unhashable-members-that-compare-equal"),
    ],
)
def test_find_still_matches_values_the_index_key_could_split(tmp_path: Path, appended: Any, queried: Any) -> None:
    """A key finer than equality hides a row, and no later check can recover it."""
    trajectory = Trajectory(iteration_dir=tmp_path)
    row = trajectory.append(step=1, action=appended, reward=1.0, observation={"metric": 1.0})

    match = trajectory.find(action=queried)

    assert match is not None
    assert match["step"] == row["step"]


def test_dataframe_keeps_an_int_in_a_column_that_also_holds_floats(tmp_path: Path) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)
    trajectory.append(step=1, action={"x": 1}, reward=1, observation={"metric": 1.0})
    trajectory.append(step=2, action={"x": 2}, reward=2.5, observation={"metric": 2.0})

    rewards = trajectory.dataframe["reward"]

    assert type(rewards.iloc[0]) is int
    assert type(rewards.iloc[1]) is float


@pytest.mark.parametrize("step", [0, True, 1.0])
def test_trajectory_rejects_invalid_steps(tmp_path: Path, step: Any) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)

    with pytest.raises(ValueError, match="positive integer"):
        trajectory.append(step=step, action={"x": 1}, reward=1.0, observation={"metric": 1.0})


def test_trajectory_rejects_non_increasing_steps(tmp_path: Path) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)

    trajectory.append(step=2, action={"x": 1}, reward=1.0, observation={"metric": 1.0})
    with pytest.raises(ValueError, match="steps must increase"):
        trajectory.append(step=2, action={"x": 2}, reward=1.0, observation={"metric": 1.0})


def test_first_row_establishes_fixed_schema(tmp_path: Path) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)
    trajectory.append(step=1, action={"x": 1}, reward=1.0, observation={"metric": 1.0})

    with pytest.raises(ValueError, match="record fields changed"):
        trajectory.append(
            step=2,
            action={"x": 2},
            reward=2.0,
            observation={"metric": 2.0},
            logging={"power": 600.0},
        )


def test_flattening_rejects_duplicate_columns(tmp_path: Path) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)

    with pytest.raises(ValueError, match=r"duplicate column: action\.x"):
        trajectory.append(
            step=1,
            action={"x": 1},
            reward=1.0,
            observation={"metric": 1.0},
            **{"action.x": 2},
        )


def test_flattening_rejects_non_string_mapping_keys(tmp_path: Path) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)

    with pytest.raises(TypeError, match="mapping keys must be strings"):
        trajectory.append(step=1, action={1: "x"}, reward=1.0, observation={"metric": 1.0})


@pytest.mark.parametrize("missing", ["action", "reward", "observation"])
def test_append_requires_core_values(tmp_path: Path, missing: str) -> None:
    values = {"action": {"x": 1}, "reward": 1.0, "observation": {"metric": 2.0}}
    del values[missing]

    with pytest.raises(TypeError, match=missing):
        Trajectory(iteration_dir=tmp_path).append(step=1, **values)

    assert not (tmp_path / "trajectory.csv").exists()


def test_warm_start_dataframe_is_copied_and_not_replayed(tmp_path: Path) -> None:
    samples = [1, 2]
    dataframe = pd.DataFrame(
        [
            {
                "step": 1,
                "action.x": 1,
                "reward": 1.0,
                "observation.metric": 2.0,
                "logging.samples": samples,
            }
        ],
        dtype=object,
    )
    trajectory = Trajectory(dataframe=dataframe, iteration_dir=tmp_path)
    dataframe.loc[0, "reward"] = 99.0
    samples.append(3)

    assert len(trajectory) == 1
    assert trajectory.dataframe.loc[0, "reward"] == 1.0
    assert trajectory.dataframe.loc[0, "logging.samples"] == [1, 2]
    assert not (tmp_path / "trajectory.csv").exists()


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([{"action.x": 1}], "must contain a step column"),
        ([{"step": 0}], "positive integer"),
        ([{"step": 2}, {"step": 1}], "steps must increase"),
    ],
)
def test_warm_start_dataframe_validation(tmp_path: Path, records: list[dict[str, object]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Trajectory(iteration_dir=tmp_path, dataframe=pd.DataFrame(records))


def test_warm_start_dataframe_rejects_duplicate_columns(tmp_path: Path) -> None:
    dataframe = pd.DataFrame([[1, 2]], columns=["step", "step"])

    with pytest.raises(ValueError, match="columns must be unique"):
        Trajectory(iteration_dir=tmp_path, dataframe=dataframe)


def test_csv_initializes_a_precreated_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    path.touch()

    Trajectory(iteration_dir=tmp_path).append(step=1, action={"x": 1}, reward=1.0, observation={"metric": 2.0})

    assert path.read_text().splitlines() == [
        "step,action,reward,observation",
        "1,{'x': 1},1.0,[2.0]",
    ]


@pytest.mark.parametrize(
    "header, row",
    [
        ("step,action,reward,observation", "1,{'x': 1},1.0,[2.0]"),
        ("step,reward,action,observation", "1,1.0,{'x': 1},[2.0]"),
    ],
)
def test_csv_reuses_a_matching_header_without_duplication(tmp_path: Path, header: str, row: str) -> None:
    path = tmp_path / "trajectory.csv"
    path.write_text(header + "\n")

    Trajectory(iteration_dir=tmp_path, mode="adopt").append(
        step=1, action={"x": 1}, reward=1.0, observation={"metric": 2.0}
    )

    assert path.read_text().splitlines() == [header, row]
    record = Trajectory.load(tmp_path).records()[0]
    assert record["action"] == {"x": 1}
    assert record["reward"] == 1.0 and record["observation"] == [2.0]


def test_csv_rejects_an_existing_mismatched_header(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    path.write_text("step,unexpected,action,observation\n")

    with pytest.raises(configurator.TrajectoryLoadError, match="missing required columns"):
        Trajectory(iteration_dir=tmp_path, mode="adopt").append(
            step=1, action={"x": 1}, reward=1.0, observation={"metric": 2.0}
        )

    assert path.read_text() == "step,unexpected,action,observation\n"


def test_core_only_trajectory_does_not_create_metadata_file(tmp_path: Path) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)

    trajectory.append(step=1, action={"x": 1}, reward=1.0, observation={"metric": 2.0})

    assert trajectory.metadata_path.read_text().splitlines() == ["step", "1"]


@pytest.mark.parametrize(
    "header, row",
    [
        ("step,env_params.speed,logging.power", "1,100,600.0"),
        ("step,logging.power,env_params.speed", "1,600.0,100"),
    ],
)
def test_metadata_reuses_matching_header_without_duplication(tmp_path: Path, header: str, row: str) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(header + "\n")
    (tmp_path / "trajectory.csv").write_text("step,action,reward,observation\n")
    trajectory = Trajectory(iteration_dir=tmp_path, mode="adopt")

    trajectory.append(
        step=1,
        action={"x": 1},
        reward=1.0,
        observation={"metric": 2.0},
        env_params={"speed": 100},
        logging={"power": 600.0},
    )

    assert path.read_text().splitlines() == [header, row]
    assert Trajectory.load(tmp_path).records()[0]["logging"] == {"power": "600.0"}


def test_metadata_initializes_a_precreated_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    path.touch()
    trajectory = Trajectory(iteration_dir=tmp_path)

    trajectory.append(
        step=1,
        action={"x": 1},
        reward=1.0,
        observation={"metric": 2.0},
        env_params={"speed": 100},
    )

    assert path.read_text().splitlines() == ["step,env_params.speed", "1,100"]


def test_metadata_rejects_an_existing_mismatched_header_before_core_write(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text("step,logging.power,env_params.other\n")
    core = tmp_path / "trajectory.csv"
    core.write_text("step,action,reward,observation\n")
    trajectory = Trajectory(iteration_dir=tmp_path, mode="adopt")
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.csv")}

    with pytest.raises(ValueError, match="file fields do not match"):
        trajectory.append(
            step=1,
            action={"x": 1},
            reward=1.0,
            observation={"metric": 2.0},
            env_params={"speed": 100},
            logging={"power": 600.0},
        )

    assert before == {path.name: path.read_bytes() for path in tmp_path.glob("*.csv")}
    assert len(trajectory) == 0


def test_persistence_failure_does_not_append_to_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("write failed")

    monkeypatch.setattr(os, "fsync", fail_write)

    with pytest.raises(OSError, match="write failed"):
        trajectory.append(step=1, action={"x": 1}, reward=1.0, observation={"metric": 2.0})

    assert len(trajectory) == 0


def test_trajectory_logs_lifecycle_and_lookup(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("DEBUG"):
        trajectory = Trajectory(iteration_dir=tmp_path)
        trajectory.append(step=1, action={"x": 1}, reward=1.0, observation={"metric": 2.0})
        assert trajectory.find(action={"x": 1}) is not None
        assert trajectory.find(action={"x": 2}) is None

    assert "Initializing Trajectory: entries=0, columns=[]." in caplog.messages
    assert "Appended trajectory row for step 1 (total rows: 1)." in caplog.messages


def test_find_agrees_with_an_exhaustive_scan(tmp_path: Path) -> None:
    """The lookup index must select exactly the row a linear exact-match scan would.

    ``find`` is backed by a hash index rather than a scan, so the index key has to carry
    the same identity semantics as ``_values_match_exact`` -- exact type, then structure.
    This asserts the two agree on every recorded row rather than on a handful of cases.
    """
    import random

    from cloudai.configurator.trajectory import _values_match_exact

    trajectory = Trajectory(iteration_dir=tmp_path)
    random.seed(0)
    recorded = []
    for step in range(1, 201):
        action = {"x": random.randint(0, 9), "y": random.choice(["a", "b"])}
        env = {"regime": random.choice([0.0, 0.5])}
        trajectory.append(step=step, action=action, reward=step / 100, observation=[float(step)], env_params=env)
        recorded.append((action, env))

    frame = trajectory.dataframe

    def scan(action: dict, env: dict) -> object | None:
        criteria = {f"action.{k}": v for k, v in action.items()}
        criteria.update({f"env_params.{k}": v for k, v in env.items()})
        for _, row in frame.iterrows():
            if all(_values_match_exact(row[field], value) for field, value in criteria.items()):
                return row["step"]
        return None

    for action, env in recorded:
        found = trajectory.find(action=action, env_params=env)
        expected = scan(action, env)
        assert expected is not None
        assert found is not None
        assert found["step"] == expected


def test_find_still_misses_an_unrecorded_combination(tmp_path: Path) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)
    trajectory.append(step=1, action={"x": 1}, reward=0.5, observation=[1.0], env_params={"regime": 0.0})

    assert trajectory.find(action={"x": 2}, env_params={"regime": 0.0}) is None
    assert trajectory.find(action={"x": 1}, env_params={"regime": 1.0}) is None


def test_find_sees_rows_appended_after_its_index_was_built(tmp_path: Path) -> None:
    """The index is built on first use, so later appends must keep it current."""
    trajectory = Trajectory(iteration_dir=tmp_path)
    trajectory.append(step=1, action={"x": 1}, reward=0.5, observation=[1.0])

    assert trajectory.find(action={"x": 1}) is not None
    assert trajectory.find(action={"x": 2}) is None

    trajectory.append(step=2, action={"x": 2}, reward=0.6, observation=[2.0])

    found = trajectory.find(action={"x": 2})
    assert found is not None
    assert found["step"] == 2


def test_find_returns_the_first_match_when_an_action_repeats(tmp_path: Path) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)
    for step in (1, 2, 3):
        trajectory.append(step=step, action={"x": 7}, reward=step / 10, observation=[float(step)])

    found = trajectory.find(action={"x": 7})
    assert found is not None
    assert found["step"] == 1


def test_lookup_cost_does_not_grow_with_trajectory_length(tmp_path: Path) -> None:
    """Guards the O(N^2) regression: per-call cost must not scale with rows already stored.

    Queries target the most recently appended rows. A linear scan short-circuits on the
    first match, so querying the front of the trajectory stays cheap however long it grows
    and would not detect the regression at all -- the cost only shows when the match is at
    the far end. The threshold is deliberately loose: this asserts a change in complexity
    class, not a timing budget.
    """
    import time

    trajectory = Trajectory(iteration_dir=tmp_path)
    for step in range(1, 201):
        trajectory.append(step=step, action={"x": step}, reward=0.1, observation=[1.0])

    start = time.perf_counter()
    for step in range(1, 201):
        trajectory.find(action={"x": step})
    small = time.perf_counter() - start

    for step in range(201, 1601):
        trajectory.append(step=step, action={"x": step}, reward=0.1, observation=[1.0])

    start = time.perf_counter()
    for step in range(1401, 1601):
        trajectory.find(action={"x": step})
    large = time.perf_counter() - start

    assert large < small * 3, f"lookup cost grew with length: {small:.4f}s -> {large:.4f}s over an 8x longer trajectory"


def test_load_reconstructs_complete_trials_from_the_real_writer(tmp_path: Path) -> None:
    written = Trajectory(iteration_dir=tmp_path)
    for step, ooo in enumerate((10, 11), start=1):
        written.append(
            step=step,
            action={"ooo": ooo},
            reward=step / 10,
            observation={"bus_bw": step * 10.0},
            env_params={"workload": "bisection", "msg_size": 1024, "drop_rate": 0.003, "network.port": 100},
            execution={"ticket": f"T-{step}"},
        )
    live = written.records()[0]
    assert live["env_params"] == {
        "workload": "bisection",
        "msg_size": 1024,
        "drop_rate": 0.003,
        "network": {"port": 100},
    }

    original_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    loaded = Trajectory.load(tmp_path)
    records = loaded.records()

    assert [record.step for record in records] == [1, 2]
    assert [record["action"] for record in records] == [{"ooo": 10}, {"ooo": 11}]
    assert [record["reward"] for record in records] == [0.1, 0.2]
    assert [record["observation"] for record in records] == [[10.0], [20.0]]
    assert records[0]["env_params"] == {
        "workload": "bisection",
        "msg_size": "1024",
        "drop_rate": "0.003",
        "network": {"port": "100"},
    }
    assert records[0]["execution"] == {"ticket": "T-1"}
    assert records[0].source == written.output_path
    assert "observation.bus_bw" not in loaded.dataframe.columns

    found = loaded.find(action={"ooo": 11})
    assert found is not None
    assert found["step"] == 2

    assert loaded.dataframe["step"].tolist() == [record.step for record in loaded.records()]
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original_files


def test_load_preserves_source_metadata_until_an_explicit_projection(tmp_path: Path) -> None:
    written = Trajectory(iteration_dir=tmp_path)
    written.append(
        step=1,
        action={"ooo": 10},
        reward=1.0,
        observation=[100.0],
        env_params={"label": "001", "msg_size": 1024, "drop_rate": 0.003},
        source_row={"label": "001"},
    )

    loaded = Trajectory.load(tmp_path)
    projected = loaded.records()[0].get("env_params", decode_numbers=True)

    assert projected == {"label": 1, "msg_size": 1024, "drop_rate": 0.003}
    assert loaded.records()[0]["env_params"] == {"label": "001", "msg_size": "1024", "drop_rate": "0.003"}
    assert loaded.records()[0].get("env_params") == loaded.records()[0]["env_params"]
    assert loaded.records()[0].get("source_row") == {"label": "001"}
    assert loaded.records()[0]["source_row"] == {"label": "001"}


def test_loaded_views_and_records_do_not_share_mutable_state(tmp_path: Path) -> None:
    written = Trajectory(iteration_dir=tmp_path)
    written.append(step=1, action={"values": [1, 2]}, reward=1.0, observation=[2.0], execution={"ticket": "T-1"})

    loaded = Trajectory.load(tmp_path)
    record = loaded.records()[0]

    record["action"]["values"].append(3)
    record["execution"]["ticket"] = "changed"

    flat = loaded.dataframe
    values: object = flat.at[0, "action.values"]
    assert isinstance(values, list)
    values.append(3)

    serialized = loaded.dataframe
    serialized.loc[0, "reward"] = 9.0

    assert loaded.records()[0]["action"] == {"values": [1, 2]}
    assert loaded.records()[0]["execution"] == {"ticket": "T-1"}

    assert loaded.dataframe.loc[0, "action.values"] == [1, 2]
    assert loaded.dataframe.loc[0, "reward"] == 1.0


@pytest.mark.parametrize("domain", ["step", "action", "reward", "observation"])
def test_inline_auxiliary_columns_cannot_change_core_values(tmp_path: Path, domain: str) -> None:
    source = tmp_path / "trajectory.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "action", "reward", "observation", f"{domain}.extra"])
        writer.writerow([1, {}, {}, {}, "auxiliary"])
    original = source.read_bytes()

    with pytest.raises(configurator.TrajectoryLoadError, match="overlaps core domain") as failure:
        Trajectory.load(source)

    assert str(source) in str(failure.value)
    assert domain in str(failure.value)
    assert source.read_bytes() == original


def test_load_accepts_an_explicit_csv_with_its_sibling_metadata(tmp_path: Path) -> None:
    written = Trajectory(iteration_dir=tmp_path)
    written.append(step=1, action={"ooo": 10}, reward=1.0, observation=[100.0], env_params={"workload": "bisection"})
    csv_path = written.output_path.rename(tmp_path / "measurements.csv")

    record = Trajectory.load(csv_path).records()[0]

    assert record.source == csv_path
    assert record["env_params"] == {"workload": "bisection"}


@pytest.mark.parametrize("with_metadata", [False, True])
def test_load_accepts_a_header_only_history(tmp_path: Path, with_metadata: bool) -> None:
    (tmp_path / "trajectory.csv").write_text("step,action,reward,observation\n")
    if with_metadata:
        (tmp_path / "metadata.csv").write_text("step,env_params.workload\n")

    loaded = Trajectory.load(tmp_path)

    assert len(loaded) == 0
    assert loaded.records() == ()
    assert loaded.dataframe.empty


@pytest.mark.parametrize(
    ("core", "metadata", "artifact"),
    [
        (
            "step,action,reward,observation\n1,{'ooo': 10},0.1,[10.0]\n2,{'ooo': 11},0.2,[20.0]\n",
            "step,env_params.workload\n1,bisection\n",
            "metadata.csv",
        ),
        (
            "step,action,reward,observation\n1,{'ooo': 10},0.1,[10.0]\n",
            "step,env_params.workload\n1,bisection\n1,many2one\n",
            "metadata.csv",
        ),
        (
            "step,action,reward,observation\n1,{'ooo': 10},0.1,[10.0]\n",
            "step,env_params.workload\n1,bisection\n2,many2one\n",
            "metadata.csv",
        ),
        (
            "step,action,reward,observation\n1,{'ooo': 10},0.1,[10.0]\n",
            "step,env_params.workload\ninvalid,bisection\n",
            "metadata.csv",
        ),
        (
            "step,action,reward,observation\n1,{'ooo': 10},0.1,[10.0]\n1,{'ooo': 11},0.2,[20.0]\n",
            None,
            "trajectory.csv",
        ),
        (
            "step,action,reward,observation\n2,{'ooo': 10},0.1,[10.0]\n1,{'ooo': 11},0.2,[20.0]\n",
            None,
            "trajectory.csv",
        ),
        ("step,action,reward,observation\n-1,{'ooo': 10},0.1,[10.0]\n", None, "trajectory.csv"),
        ("step,action,reward,observation\ninvalid,{'ooo': 10},0.1,[10.0]\n", None, "trajectory.csv"),
        ("step,reward,observation\n1,0.1,[10.0]\n", None, "trajectory.csv"),
        ("step,action,reward,reward,observation\n1,{'ooo': 10},0.1,0.2,[10.0]\n", None, "trajectory.csv"),
        (
            "step,action,reward,observation\n1,{'ooo': 10},0.1,[10.0]\n",
            "step,env_params.workload,env_params.workload\n1,bisection,many2one\n",
            "metadata.csv",
        ),
        (
            "step,action,reward,observation\n1,{'ooo': 10},0.1,[10.0]\n",
            "env_params.workload\nbisection\n",
            "metadata.csv",
        ),
        ("step,action,reward,observation\n1,{'ooo': 10},0.1\n", None, "trajectory.csv"),
        ("step,action,reward,observation\n1,{'ooo': 10},0.1,[10.0],extra\n", None, "trajectory.csv"),
        ('step,action,reward,observation\n1,"unterminated\n', None, "trajectory.csv"),
        pytest.param(
            "step,action,reward,observation\n1,{},1,[]\n",
            "step,execution..ticket\n1,T\n",
            "metadata.csv",
            id="empty-metadata-path",
        ),
        pytest.param(
            "step,action,reward,observation\n1,{},1,[]\n",
            "step,execution.ticket,execution.ticket.detail\n1,T,D\n",
            "metadata.csv",
            id="scalar-metadata-path",
        ),
        pytest.param(
            "step,action,reward,observation\n1,{},1,[]\n",
            "step,execution.ticket.detail,execution.ticket\n1,D,T\n",
            "metadata.csv",
            id="reverse-metadata-path",
        ),
        pytest.param(
            "step,action,reward,observation,execution.ticket\n1,{},1,[],inline\n",
            "step,execution.ticket\n1,sidecar\n",
            "metadata.csv",
            id="inline-sidecar-duplicate",
        ),
    ],
)
def test_load_rejects_inconsistent_artifacts_before_returning_history(
    tmp_path: Path, core: str, metadata: str | None, artifact: str
) -> None:
    (tmp_path / "trajectory.csv").write_text(core)
    if metadata is not None:
        (tmp_path / "metadata.csv").write_text(metadata)
    original_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    with pytest.raises(configurator.TrajectoryLoadError, match=artifact) as failure:
        Trajectory.load(tmp_path)

    assert str(failure.value).startswith(f"Cannot load trajectory {tmp_path / artifact}:")
    assert "row " in str(failure.value) or "step " in str(failure.value)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original_files


@pytest.mark.parametrize("artifact", ["trajectory.csv", "metadata.csv"])
def test_load_reports_unreadable_or_undecodable_artifact(tmp_path: Path, artifact: str) -> None:
    (tmp_path / "trajectory.csv").write_text("step,action,reward,observation\n1,{'ooo': 10},0.1,[10.0]\n")
    (tmp_path / artifact).write_bytes(b"\xff\xfe")

    with pytest.raises(configurator.TrajectoryLoadError, match=artifact):
        Trajectory.load(tmp_path)


def test_load_reports_missing_required_artifact(tmp_path: Path) -> None:
    with pytest.raises(configurator.TrajectoryLoadError, match=r"trajectory\.csv"):
        Trajectory.load(tmp_path)


def test_load_identifies_the_step_whose_metadata_is_missing(tmp_path: Path) -> None:
    written = Trajectory(iteration_dir=tmp_path)
    for step in (1, 2):
        written.append(
            step=step, action={"ooo": step}, reward=1.0, observation=[100.0], env_params={"workload": "bisection"}
        )
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "trajectory.csv").write_bytes(written.output_path.read_bytes())
    metadata = legacy / "metadata.csv"
    metadata.write_text("step,env_params.workload\n1,bisection\n")

    with pytest.raises(configurator.TrajectoryLoadError) as failure:
        Trajectory.load(legacy)

    assert str(metadata) in str(failure.value)
    assert "step 2" in str(failure.value)
    assert "missing metadata row for trajectory step" in str(failure.value)


def test_load_accepts_an_extensionless_csv_path(tmp_path: Path) -> None:
    written = Trajectory(iteration_dir=tmp_path)
    written.append(step=1, action={"ooo": 10}, reward=1.0, observation=[100.0])
    source = written.output_path.rename(tmp_path / "history")

    record = Trajectory.load(source).records()[0]

    assert record.source == source
    assert record["action"] == {"ooo": 10}


def test_load_rejects_a_history_without_an_observation_column(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    path.write_text("step,action,reward\n1,{'ooo': 10},0.1\n")
    original = path.read_bytes()

    with pytest.raises(configurator.TrajectoryLoadError) as failure:
        Trajectory.load(tmp_path)

    assert str(failure.value) == f"Cannot load trajectory {path}: row 1, missing required columns: ['observation']"
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("cell", "expected"),
    [("", ""), ("None", None), ("[]", []), ("[truncated observation]", "[truncated observation]")],
)
def test_load_preserves_present_observation_cells(tmp_path: Path, cell: str, expected: Any) -> None:
    path = tmp_path / "trajectory.csv"
    path.write_text("step,action,reward,observation\n1,{'ooo': 10},-1.0," + cell + "\n")
    original = path.read_bytes()

    record = Trajectory.load(tmp_path).records()[0]

    assert record["observation"] == expected
    assert record.get("observation", "fallback") == expected
    assert record.get("observation", "fallback", decode_numbers=True) == expected
    assert record["reward"] == -1.0
    assert path.read_bytes() == original


def test_load_accepts_large_writer_fields_and_restores_csv_reader_state(tmp_path: Path) -> None:
    payload = "x" * 140000
    written = Trajectory(iteration_dir=tmp_path)
    written.append(step=1, action={"payload": payload}, reward=1.0, observation=[payload], logging={"payload": payload})
    original_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(4096)

        loaded = Trajectory.load(tmp_path)

        assert loaded.records()[0]["action"] == {"payload": payload}
        assert loaded.records()[0]["observation"] == [payload]
        assert loaded.records()[0]["logging"] == {"payload": payload}
        assert loaded.dataframe["step"].tolist() == [record.step for record in loaded.records()]
        assert csv.field_size_limit() == 4096

        markerless = tmp_path / "markerless"
        markerless.mkdir()
        (markerless / Trajectory.file_name).write_bytes(written.output_path.read_bytes())
        (markerless / Trajectory.metadata_file_name).write_text(f'step,logging.payload\n1,"{payload}",extra\n')

        with pytest.raises(configurator.TrajectoryLoadError, match="row 2, expected 2 fields, got 3"):
            Trajectory.load(markerless)

        assert csv.field_size_limit() == 4096
    finally:
        csv.field_size_limit(original_limit)


def test_load_preserves_nested_nonfinite_numbers_from_the_real_writer(tmp_path: Path) -> None:
    values = [float("nan"), float("inf"), float("-inf")]
    written = Trajectory(iteration_dir=tmp_path)
    written.append(step=1, action={"values": values}, reward={"values": values}, observation=values)

    record = Trajectory.load(tmp_path).records()[0]

    for decoded in (record["action"]["values"], record["reward"]["values"], record["observation"]):
        assert math.isnan(decoded[0])
        assert decoded[1:] == [float("inf"), float("-inf")]


@pytest.mark.parametrize("expression", ["[unknown]", "[float('nan')]", "[nan.real]"])
def test_load_rejects_expressions_outside_the_literal_contract(tmp_path: Path, expression: str) -> None:
    with (tmp_path / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "action", "reward", "observation"])
        writer.writerow([1, {}, 1.0, expression])

    with pytest.raises(configurator.TrajectoryLoadError):
        Trajectory.load(tmp_path)


def test_load_preserves_metadata_cell_line_endings_from_the_writer(tmp_path: Path) -> None:
    note = "line1\r\nline2"
    written = Trajectory(iteration_dir=tmp_path)
    written.append(step=1, action={"ooo": 10}, reward=1.0, observation=[100.0], env_params={"note": note})
    original_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    loaded = Trajectory.load(tmp_path)

    assert loaded.records()[0]["env_params"] == {"note": note}
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original_files


def test_load_preserves_zero_based_legacy_history_and_metadata_identity(tmp_path: Path) -> None:
    core_path = tmp_path / "trajectory.csv"
    metadata_path = tmp_path / "metadata.csv"

    with core_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "action", "reward", "observation", "legacy.extra"])
        writer.writerow([0, {"ooo": 10}, 1.0, [100.0], "legacy-0"])
        writer.writerow([1, {"ooo": 11}, 2.0, [200.0], "legacy-1"])

    with metadata_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "env_params.workload"])
        writer.writerow([0, "thin"])
        writer.writerow([1, "wide"])

    original_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    loaded = Trajectory.load(tmp_path)

    records = loaded.records()

    assert [(record.step, record["env_params"]) for record in records] == [
        (0, {"workload": "thin"}),
        (1, {"workload": "wide"}),
    ]
    assert records[0]["action"] == {"ooo": 10}
    for record in records:
        assert list(record).count("step") == 1
        assert len(record) == len(set(record)) == len(dict(record))
        assert list(record.items()).count(("step", record.step)) == 1
    assert [record.get("step", decode_numbers=True) for record in records] == [0, 1]
    assert loaded.dataframe["legacy.extra"].tolist() == ["legacy-0", "legacy-1"]

    found = loaded.find(action={"ooo": 10}, env_params={"workload": "thin"})
    assert found is not None
    assert found["step"] == 0

    assert loaded.dataframe["step"].tolist() == [record.step for record in loaded.records()]
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original_files


@pytest.fixture(params=[False, True], ids=["live", "loaded"])
def readonly_record(request: pytest.FixtureRequest, tmp_path: Path):
    expected = {
        "step": 1,
        "action": {"values": ["1", "2.5", "auto"], "label": "001"},
        "reward": 0.5,
        "observation": [2.0],
        "env_params": {"message_size": "1024"},
        "execution": {"ticket": "T-1"},
        "source": {"archive": "prior"},
    }

    def create():
        trajectory = Trajectory(iteration_dir=tmp_path)
        trajectory.append(**expected)
        if request.param:
            trajectory = Trajectory.load(tmp_path)

        return trajectory.records()[0], MappingProxyType(expected)

    return create


def test_record_reads_match_a_readonly_dictionary(readonly_record, tmp_path: Path) -> None:
    record, expected = readonly_record()

    assert record.source == tmp_path / "trajectory.csv"
    assert record["source"] == {"archive": "prior"}
    assert dict(record) == {**record} == expected
    assert record == expected and expected == record
    assert list(record) == list(record.keys()) == list(expected)
    assert list(record.items()) == list(expected.items())
    assert list(record.values()) == list(expected.values())
    assert len(record) == len(expected) and bool(record) == bool(expected)
    for key in expected:
        assert key in record and record[key] == record.get(key) == expected[key]
    for missing in ("missing", "row"):
        assert missing not in record and record.get(missing) is None
        assert record.get(missing, "fallback") == "fallback"
        with pytest.raises(KeyError, match=missing):
            _ = record[missing]

    assert record.get("env_params", decode_numbers=True) == {"message_size": 1024}
    assert record.get("action", decode_numbers=True) == {"values": [1, 2.5, "auto"], "label": 1}
    assert record["env_params"] == {"message_size": "1024"}

    fallback = {"label": "001", "values": ["2"]}
    assert record.get("missing", fallback) is fallback
    assert record.get("missing", fallback, decode_numbers=True) is fallback
    assert fallback == {"label": "001", "values": ["2"]}
    assert record.get("missing", "001", decode_numbers=True) == "001"
    assert record.get("missing", decode_numbers=True) is None
    assert record.get("step", decode_numbers=True) == record.step
    assert record.get("env_params", decode_numbers=False) == record["env_params"]


def test_record_rejects_writes_and_returns_detached_nested_values(
    readonly_record,
) -> None:
    record, expected = readonly_record()
    writable: Any = record

    with pytest.raises(TypeError):
        writable["reward"] = 7.0
    with pytest.raises(TypeError):
        del writable["reward"]
    for mutator in ("clear", "pop", "popitem", "setdefault", "update"):
        assert not hasattr(record, mutator)

    for detached in (
        record["action"],
        record.get("action"),
        record.get("action", decode_numbers=True),
        dict(record.items())["action"],
        list(record.values())[1],
        dict(record)["action"],
        {**record}["action"],
    ):
        assert isinstance(detached, dict)
        detached["values"].append(3)
        assert record == expected


@pytest.mark.parametrize("source_kind", ["directory", "file"])
@pytest.mark.parametrize("artifact", ["absent", "metadata-only", "partial", "corrupt-marker", "broken-metadata"])
def test_optional_history_is_empty_only_when_all_artifacts_are_absent(
    tmp_path: Path, source_kind: str, artifact: str
) -> None:
    source = tmp_path if source_kind == "directory" else tmp_path / "trajectory.csv"
    if artifact == "metadata-only":
        (tmp_path / "metadata.csv").write_text("step\n1\n")
    elif artifact == "partial":
        (tmp_path / "trajectory.csv").write_text("step,action,reward,observation\n1,")
    elif artifact == "corrupt-marker":
        (tmp_path / ".trajectory.commit.json").write_text("{")
    elif artifact == "broken-metadata":
        (tmp_path / "metadata.csv").symlink_to(tmp_path / "missing.csv")
    before = sorted(path.name for path in tmp_path.iterdir())

    if artifact == "absent":
        with pytest.raises(configurator.TrajectoryLoadError):
            Trajectory.load(source)
        loaded = Trajectory.load(source, missing_ok=True)
        assert loaded.records() == ()
        assert loaded.dataframe.empty
        with pytest.raises(RuntimeError, match="read-only"):
            loaded.append(step=1, action={}, reward=1.0, observation=[])
    else:
        with pytest.raises(configurator.TrajectoryLoadError):
            Trajectory.load(source, missing_ok=True)
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_concurrent_large_csv_loads_restore_process_field_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread

    directories = [tmp_path / "first", tmp_path / "second"]
    for directory in directories:
        directory.mkdir()
        with (directory / Trajectory.file_name).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["step", "action", "reward", "observation", "note"])
            writer.writerow([1, {"x": 1}, 10, [1.0], "x" * 200000])
    reader = csv.reader
    first_inside, second_started, second_inside, first_finished = Event(), Event(), Event(), Event()
    ambient_limit = csv.field_size_limit()
    baseline_limit = 64

    def scheduled_reader(*args, **kwargs):
        if current_thread().name.endswith("_0"):
            first_inside.set()
            assert second_started.wait(5)
            second_inside.wait(1)
        else:
            second_inside.set()
            assert first_finished.wait(5)
        return reader(*args, **kwargs)

    def load(index: int):
        if index:
            second_started.set()
        try:
            return [record.step for record in Trajectory.load(directories[index]).records()]
        finally:
            if not index:
                first_finished.set()

    try:
        csv.field_size_limit(baseline_limit)
        with monkeypatch.context() as scheduling:
            scheduling.setattr(csv, "reader", scheduled_reader)
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="trajectory-reader") as pool:
                first = pool.submit(load, 0)
                assert first_inside.wait(5)
                second = pool.submit(load, 1)
                assert first.result(timeout=10) == [1]
                assert second.result(timeout=10) == [1]
        assert csv.field_size_limit() == baseline_limit
    finally:
        csv.field_size_limit(ambient_limit)


@pytest.mark.parametrize(
    ("action", "reward", "observation"),
    [(7, {"objective": 1.0}, (1, 2)), ({"ooo": 1}, -1.0, [-1.0]), ({"ooo": 2}, 0.0, 2.0)],
)
def test_supported_legacy_core_values_do_not_require_metadata(tmp_path, action, reward, observation):
    source = tmp_path / "trajectory.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "action", "reward", "observation"])
        writer.writerow([1, repr(action), repr(reward), repr(observation)])
    before = source.read_bytes()
    record = Trajectory.load(source).records()[0]
    assert record["action"] == action
    assert record["reward"] == reward
    assert record["observation"] == observation
    assert set(record) == {"step", "action", "reward", "observation"}
    assert source.read_bytes() == before
    assert not (tmp_path / "metadata.csv").exists()


@pytest.mark.parametrize(
    ("action", "reward", "observation", "serialized_observation", "next_observation"),
    [
        ({}, {"objective": {}}, {}, [], []),
        (
            {"cmd_args.ooo_thresh": 20, "nested": {"empty": {}}},
            {"metrics.score": 1.0, "empty": {}},
            [100.0],
            [100.0],
            [200.0],
        ),
    ],
)
def test_complete_core_records_survive_loading_and_appending(
    tmp_path: Path,
    action: Any,
    reward: Any,
    observation: Any,
    serialized_observation: list[float],
    next_observation: list[float],
) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)
    trajectory.append(step=1, action=action, reward=reward, observation=observation)
    live = trajectory.records()[0]

    assert live["action"] == action
    assert live["reward"] == reward
    assert live["observation"] == observation
    assert live.row is None

    loaded = Trajectory.load(tmp_path)
    first = loaded.records()[0]
    assert first["action"] == action
    assert first["reward"] == reward
    assert first["observation"] == serialized_observation
    assert first.row == 2

    output_path = trajectory.output_path
    del trajectory
    gc.collect()
    loaded = Trajectory(iteration_dir=tmp_path, mode="resume")
    loaded.append(step=2, action=action, reward=reward, observation=next_observation)
    records = loaded.records()
    assert [record["action"] for record in records] == [action, action]
    assert [record["reward"] for record in records] == [reward, reward]
    assert [record["observation"] for record in records] == [serialized_observation, next_observation]
    assert [record.row for record in records] == [2, None]
    assert records[0].source == records[1].source == output_path
    assert live["action"] == first["action"] == records[0]["action"]

    reloaded = Trajectory.load(tmp_path).records()
    assert [dict(record) for record in reloaded] == [dict(record) for record in records]
    assert [record.row for record in reloaded] == [2, 3]


@pytest.mark.parametrize(
    ("action", "reward", "observation"),
    [
        (7, {"objective": 1.0}, (1, 2)),
        ({"ooo": 1}, -1.0, [-1.0]),
        ({"ooo": 2}, 0.0, 2.0),
    ],
)
def test_load_accepts_writer_supported_domains_without_optional_metadata(
    tmp_path: Path, action: Any, reward: Any, observation: Any
) -> None:
    written = Trajectory(iteration_dir=tmp_path)
    written.append(step=1, action=action, reward=reward, observation=observation)

    live_record = written.records()[0]
    record = Trajectory.load(tmp_path).records()[0]

    assert record["action"] == action
    assert live_record["action"] == record["action"]
    assert record["reward"] == reward
    assert live_record["reward"] == record["reward"]
    assert record["observation"] == observation
    assert live_record["observation"] == record["observation"]
    assert set(record) == {"step", "action", "reward", "observation"}
    assert written.metadata_path.read_text().splitlines() == ["step", "1"]
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "trajectory.csv").write_bytes(written.output_path.read_bytes())
    assert dict(Trajectory.load(legacy).records()[0]) == dict(record)
    assert not (legacy / "metadata.csv").exists()


def test_loading_history_remains_readonly_while_writer_owns_it(tmp_path: Path) -> None:
    owner = Trajectory(iteration_dir=tmp_path)
    owner.append(step=1, action={"x": 1}, reward=1.0, observation=[])
    loaded = Trajectory.load(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.suffix == ".csv"}

    with pytest.raises((ValueError, RuntimeError)):
        loaded.append(step=2, action={"x": 2}, reward=2.0, observation=[])

    assert [record.step for record in loaded.records()] == [1]
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.suffix == ".csv"}


def test_second_writer_cannot_change_an_owned_history(tmp_path: Path) -> None:
    owner = Trajectory(iteration_dir=tmp_path)
    owner.append(step=1, action={"x": 1}, reward=1.0, observation=[])
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.suffix == ".csv"}

    with pytest.raises((OSError, ValueError, RuntimeError)):
        Trajectory(iteration_dir=tmp_path, mode="resume").append(step=2, action={"x": 2}, reward=2.0, observation=[])

    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.suffix == ".csv"}
    assert [record.step for record in Trajectory.load(tmp_path).records()] == [1]


@pytest.mark.parametrize("with_metadata", [False, True])
@pytest.mark.parametrize("reordered", [False, True])
def test_create_and_explicit_adoption_preserve_legacy_history(
    tmp_path: Path, with_metadata: bool, reordered: bool
) -> None:
    core = tmp_path / "trajectory.csv"
    metadata = tmp_path / "metadata.csv"
    core.write_text("step,action,reward,observation\n0,{},-1.0,[]\n3,{},2.0,[]\n")
    if with_metadata:
        metadata.write_text("step,env_params.label\n0,failed\n3,succeeded\n")
    if reordered:
        core.write_text("observation,reward,action,step\n[],-1.0,{},0\n[],2.0,{},3\n")
        if with_metadata:
            metadata.write_text("env_params.label,step\nfailed,0\nsucceeded,3\n")
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.csv")}

    with pytest.raises((OSError, ValueError, RuntimeError)):
        Trajectory(iteration_dir=tmp_path).append(step=4, action={}, reward=3.0, observation=[])
    assert before == {path.name: path.read_bytes() for path in tmp_path.glob("*.csv")}

    with pytest.raises((OSError, ValueError, RuntimeError)):
        Trajectory(iteration_dir=tmp_path, mode="resume")
    assert before == {path.name: path.read_bytes() for path in tmp_path.glob("*.csv")}

    adopted = Trajectory(iteration_dir=tmp_path, mode="adopt")
    assert [record.step for record in adopted.records()] == [0, 3]
    assert before == {name: (tmp_path / name).read_bytes() for name in before}
    extra = {"env_params": {"label": "continued"}} if with_metadata else {}
    adopted.append(step=4, action={}, reward=3.0, observation=[], **extra)
    del adopted
    gc.collect()
    resumed = Trajectory(iteration_dir=tmp_path, mode="resume")
    assert [record.step for record in resumed.records()] == [0, 3, 4]
    assert [record["reward"] for record in Trajectory.load(tmp_path).records()] == [-1.0, 2.0, 3.0]
    assert all((tmp_path / name).read_bytes().startswith(content) for name, content in before.items())


def test_readers_ignore_unpublished_tails_and_resume_removes_only_those_tails(tmp_path: Path) -> None:
    owner = Trajectory(iteration_dir=tmp_path)
    owner.append(step=1, action={"x": 1}, reward=1.0, observation=[], env_params={"label": "first"})
    committed = {path.name: path.read_bytes() for path in (owner.output_path, owner.metadata_path)}
    with owner.output_path.open("a", newline="") as stream:
        csv.writer(stream).writerow([2, {"x": 2}, 2.0, []])
    with owner.metadata_path.open("a", newline="") as stream:
        csv.writer(stream).writerow([2, "unpublished"])

    assert [record.step for record in Trajectory.load(tmp_path).records()] == [1]
    del owner
    gc.collect()
    resumed = Trajectory(iteration_dir=tmp_path, mode="resume")
    assert committed == {path.name: path.read_bytes() for path in (resumed.output_path, resumed.metadata_path)}
    resumed.append(step=2, action={"x": 2}, reward=2.0, observation=[], env_params={"label": "second"})
    assert [record.step for record in Trajectory.load(tmp_path).records()] == [1, 2]


@pytest.mark.parametrize("failure_point", ["trajectory", "metadata", "marker", "directory"])
def test_retry_after_sync_failure_exposes_only_a_committed_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    owner = Trajectory(iteration_dir=tmp_path)
    owner.append(step=1, action={"x": 1}, reward=1.0, observation=[], env_params={"label": "first"})
    marker = tmp_path / ".trajectory.commit.json"
    previous_marker = marker.read_bytes() if marker.exists() else None
    fsync = os.fsync
    failed = False

    def fail_sync(descriptor: int) -> None:
        nonlocal failed
        descriptor_stat = os.fstat(descriptor)
        identity = descriptor_stat.st_dev, descriptor_stat.st_ino
        if stat.S_ISDIR(descriptor_stat.st_mode):
            kind = "directory"
        elif identity == (owner.output_path.stat().st_dev, owner.output_path.stat().st_ino):
            kind = "trajectory"
        elif identity == (owner.metadata_path.stat().st_dev, owner.metadata_path.stat().st_ino):
            kind = "metadata"
        else:
            kind = "marker"
        published = marker.exists() and marker.read_bytes() != previous_marker
        if kind == failure_point and not failed and (kind != "directory" or published):
            failed = True
            visible = [1, 2] if failure_point == "directory" else [1]
            assert [record.step for record in Trajectory.load(tmp_path).records()] == visible
            raise OSError(f"injected {failure_point} sync failure")
        fsync(descriptor)

    with monkeypatch.context() as fault:
        fault.setattr(os, "fsync", fail_sync)
        with pytest.raises(OSError, match=f"injected {failure_point} sync failure"):
            owner.append(step=2, action={"x": 2}, reward=2.0, observation=[], env_params={"label": "second"})

    assert failed
    assert len(owner) == 1
    visible = [1, 2] if failure_point == "directory" else [1]
    assert [record.step for record in Trajectory.load(tmp_path).records()] == visible
    owner.append(step=2, action={"x": 2}, reward=2.0, observation=[], env_params={"label": "second"})
    assert len(owner) == 2
    assert [record.step for record in Trajectory.load(tmp_path).records()] == [1, 2]


def test_publication_failure_after_marker_replace_retries_without_duplicate_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = Trajectory(iteration_dir=tmp_path)
    owner.append(step=1, action={"x": 1}, reward=1.0, observation=[], env_params={"label": "first"})
    replace = os.replace
    failed = False

    def replace_then_fail(source: str | Path, destination: str | Path) -> None:
        nonlocal failed
        replace(source, destination)
        if Path(destination).name == ".trajectory.commit.json" and not failed:
            failed = True
            raise OSError("injected lost acknowledgement")

    with monkeypatch.context() as fault:
        fault.setattr(os, "replace", replace_then_fail)
        with pytest.raises(OSError, match="injected lost acknowledgement"):
            owner.append(step=2, action={"x": 2}, reward=2.0, observation=[], env_params={"label": "second"})

    assert failed
    assert [record.step for record in Trajectory.load(tmp_path).records()] == [1, 2]
    committed = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    with pytest.raises((ValueError, RuntimeError)):
        owner.append(step=2, action={"x": 99}, reward=2.0, observation=[], env_params={"label": "second"})
    assert committed == {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    retried = owner.append(step=2, action={"x": 2}, reward=2.0, observation=[], env_params={"label": "second"})
    assert retried["reward"] == 2.0
    assert [record.step for record in owner.records()] == [1, 2]
    assert [record.step for record in Trajectory.load(tmp_path).records()] == [1, 2]


def test_failed_empty_marker_publication_precedes_first_csv_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace = os.replace
    failed = False

    def refuse_empty_marker(source: str | Path, destination: str | Path) -> None:
        nonlocal failed
        if Path(destination).name == ".trajectory.commit.json":
            failed = True
            for path in tmp_path.glob("*.csv"):
                with path.open(newline="") as stream:
                    assert list(csv.DictReader(stream)) == []
            raise OSError("injected empty marker publication failure")
        replace(source, destination)

    with monkeypatch.context() as fault:
        fault.setattr(os, "replace", refuse_empty_marker)
        with pytest.raises(OSError, match="injected empty marker publication failure"):
            Trajectory(iteration_dir=tmp_path).append(step=1, action={}, reward=1.0, observation=[])

    assert failed
    for path in tmp_path.glob("*.csv"):
        with path.open(newline="") as stream:
            assert list(csv.DictReader(stream)) == []
    owner = Trajectory(iteration_dir=tmp_path)
    owner.append(step=1, action={}, reward=1.0, observation=[])
    assert [record.step for record in Trajectory.load(tmp_path).records()] == [1]


@pytest.mark.parametrize(
    "field, value",
    [
        (("format_version",), 99),
        (("format_version",), True),
        (("history_id",), "not-a-history-identity"),
        (("history_id",), "00000000-0000-4000-8000-000000000002"),
        (("generation",), True),
        (("generation",), 2),
        (("last_step",), 2),
        (("trajectory", "bytes"), -1),
        (("trajectory", "bytes"), "beyond-eof"),
        (("trajectory", "bytes"), 2**128),
        (("trajectory", "sha256"), "0" * 64),
        (("metadata", "bytes"), 0),
        (("last_record_sha256",), "0" * 64),
    ],
)
def test_corrupt_commit_markers_never_fall_back_to_unbounded_csv_loading(
    tmp_path: Path, field: tuple[str, ...], value: Any
) -> None:
    owner = Trajectory(iteration_dir=tmp_path)
    owner.append(step=1, action={"x": 1}, reward=1.0, observation=[], env_params={"label": "first"})
    del owner
    gc.collect()
    marker_path = tmp_path / ".trajectory.commit.json"
    marker = json.loads(marker_path.read_text())
    target = marker
    for component in field[:-1]:
        target = target[component]
    target[field[-1]] = (tmp_path / "trajectory.csv").stat().st_size + 1 if value == "beyond-eof" else value
    marker_path.write_text(json.dumps(marker))
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.csv")}

    with pytest.raises(configurator.TrajectoryLoadError):
        Trajectory.load(tmp_path)
    with pytest.raises((ValueError, configurator.TrajectoryLoadError)):
        Trajectory(iteration_dir=tmp_path, mode="resume")

    assert before == {path.name: path.read_bytes() for path in tmp_path.glob("*.csv")}


@pytest.mark.parametrize("damage", ["missing-metadata", "short-core", "bounded-partial-row"])
def test_managed_prefixes_require_all_committed_bytes_and_complete_rows(tmp_path: Path, damage: str) -> None:
    owner = Trajectory(iteration_dir=tmp_path)
    owner.append(step=1, action={"x": 1}, reward=1.0, observation=[], env_params={"label": "first"})
    del owner
    gc.collect()
    marker_path = tmp_path / ".trajectory.commit.json"
    marker = json.loads(marker_path.read_text())
    core = tmp_path / "trajectory.csv"
    if damage == "missing-metadata":
        (tmp_path / "metadata.csv").unlink()
    elif damage == "short-core":
        core.write_bytes(core.read_bytes()[:-1])
    else:
        contents = core.read_bytes()
        prefix = contents[: len(contents.splitlines(keepends=True)[0]) + 3]
        marker["trajectory"] = {
            "bytes": len(prefix),
            "sha256": hashlib.sha256(marker["history_id"].encode() + b"\x00" + prefix).hexdigest(),
        }
        marker_path.write_text(json.dumps(marker))

    with pytest.raises(configurator.TrajectoryLoadError):
        Trajectory.load(tmp_path)


@pytest.mark.parametrize("contents", ["{", "{}"])
def test_malformed_commit_marker_is_an_error_even_when_csvs_are_valid(tmp_path: Path, contents: str) -> None:
    owner = Trajectory(iteration_dir=tmp_path)
    owner.append(step=1, action={}, reward=1.0, observation=[])
    (tmp_path / ".trajectory.commit.json").write_text(contents)

    with pytest.raises(configurator.TrajectoryLoadError):
        Trajectory.load(tmp_path)


@pytest.mark.parametrize("precreate_csvs", [False, True])
def test_empty_committed_boundary_loads_and_resumes_without_csv_rows(tmp_path: Path, precreate_csvs: bool) -> None:
    history_id = "00000000-0000-4000-8000-000000000001"
    empty_hash = hashlib.sha256(history_id.encode() + b"\x00").hexdigest()
    marker = {
        "format_version": 1,
        "history_id": history_id,
        "generation": 0,
        "last_step": None,
        "trajectory": {"bytes": 0, "sha256": empty_hash},
        "metadata": {"bytes": 0, "sha256": empty_hash},
        "last_record_sha256": None,
    }
    (tmp_path / ".trajectory.commit.json").write_text(json.dumps(marker))
    if precreate_csvs:
        (tmp_path / "trajectory.csv").touch()
        (tmp_path / "metadata.csv").touch()
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    assert Trajectory.load(tmp_path).records() == ()
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    owner = Trajectory(iteration_dir=tmp_path, mode="resume")
    owner.append(step=1, action={}, reward=-1.0, observation=[float("nan")])
    record = Trajectory.load(tmp_path).records()[0]
    assert record.step == 1 and record["reward"] == -1.0
    assert math.isnan(record["observation"][0])


def test_heterogeneous_actions_keep_missing_fields_distinct_from_none_after_reload(tmp_path: Path) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)
    actions = [{"x": None}, {"y": 2}, {"x": 3, "y": None}]
    for step, action in zip((1, 2, 5), actions, strict=True):
        trajectory.append(step=step, action=action, reward=1.0, observation=[], env_params={"label": "same"})

    for history in (trajectory, Trajectory.load(tmp_path)):
        assert [record["action"] for record in history.records()] == actions
        x_none = history.find(action={"x": None})
        y_none = history.find(action={"y": None})
        assert x_none is not None and x_none["step"] == 1
        assert y_none is not None and y_none["step"] == 5
        assert history.find(action={"x": None, "y": None}) is None


def test_explicit_adoption_rejects_invalid_legacy_pair_before_creating_a_marker(tmp_path: Path) -> None:
    (tmp_path / "trajectory.csv").write_text("step,action,reward,observation\n1,{},1.0,[]\n")
    (tmp_path / "metadata.csv").write_text("step,env_params.label\n2,unmatched\n")
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.csv")}

    with pytest.raises(configurator.TrajectoryLoadError):
        Trajectory(iteration_dir=tmp_path, mode="adopt")

    assert before == {path.name: path.read_bytes() for path in tmp_path.glob("*.csv")}
    assert not (tmp_path / ".trajectory.commit.json").exists()


@pytest.mark.parametrize("inline_column", ["note", "env_params.foo"])
@pytest.mark.parametrize("with_metadata", [False, True])
def test_readable_inline_columns_cannot_be_adopted_for_writing(
    tmp_path: Path, inline_column: str, with_metadata: bool
) -> None:
    core = tmp_path / "trajectory.csv"
    core.write_text(f"step,action,reward,observation,{inline_column}\n1,{{}},1,[],old\n")
    if with_metadata:
        (tmp_path / "metadata.csv").write_text("step,env_params.sidecar\n1,retained\n")
    paths = (core, tmp_path / "metadata.csv", tmp_path / ".trajectory.commit.json")
    before = {path.name: path.read_bytes() if path.exists() else None for path in paths}
    loaded = Trajectory.load(tmp_path)
    assert loaded.dataframe.loc[0, inline_column] == "old"
    with pytest.raises(RuntimeError, match="read-only"):
        loaded.append(step=2, action={}, reward=2, observation=[])

    for _ in range(2):
        with pytest.raises(configurator.TrajectoryLoadError, match="writ") as failure:
            Trajectory(iteration_dir=tmp_path, mode="adopt")
        assert str(core) in str(failure.value) and inline_column in str(failure.value)
        assert {path.name: path.read_bytes() if path.exists() else None for path in paths} == before
        assert Trajectory.load(tmp_path).dataframe.loc[0, inline_column] == "old"


@pytest.mark.parametrize(
    "inline_column,history_id,core_digest,metadata_digest",
    [
        (
            "note",
            "cfe58403-ff40-4ec3-a7cc-696214414fe7",
            "518b072315a7d978200f2d330cce6bf0eaa3cb0b3edfad210c964fd29b07eef7",
            "e0d2afb5eb392f48233918d897c038d61c4fd180680da8aed997cc457b5dd52f",
        ),
        (
            "env_params.foo",
            "8ac2dc2e-01cb-49cf-b5f0-cd30fcc181da",
            "bab28f33aa5dd7029b3689e62eef92c402bc473c9807570862d6e4ce1efec1c1",
            "00f2b04214b5cf39120ba94b7033d9f1e70681607973053ac1c708f7e3a4f810",
        ),
    ],
)
def test_old_inline_managed_history_refuses_resume_before_recovering_tails(
    tmp_path: Path, inline_column: str, history_id: str, core_digest: str, metadata_digest: str
) -> None:
    core = tmp_path / "trajectory.csv"
    core.write_text(f"step,action,reward,observation,{inline_column}\n1,{{}},1,[],old\n")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("step\n1\n")
    marker = {
        "format_version": 1,
        "generation": 1,
        "history_id": history_id,
        "last_record_sha256": "0d9d3b9795250568ef6840868ec19c728f63c77ed25e023f8d3dc417af650261",
        "last_step": 1,
        "metadata": {"bytes": len(metadata.read_bytes()), "sha256": metadata_digest},
        "trajectory": {"bytes": len(core.read_bytes()), "sha256": core_digest},
    }
    marker_path = tmp_path / ".trajectory.commit.json"
    marker_path.write_text(json.dumps(marker))
    with core.open("ab") as stream:
        stream.write(b"2,{},2,[],unpublished\n")
    with metadata.open("ab") as stream:
        stream.write(b"2\n")
    paths = (core, metadata, marker_path)
    before = {path.name: path.read_bytes() for path in paths}
    assert Trajectory.load(tmp_path).dataframe[inline_column].tolist() == ["old"]

    for _ in range(2):
        with pytest.raises(configurator.TrajectoryLoadError, match="writ") as failure:
            Trajectory(iteration_dir=tmp_path, mode="resume")
        assert str(core) in str(failure.value) and inline_column in str(failure.value)
        assert {path.name: path.read_bytes() for path in paths} == before
        assert Trajectory.load(tmp_path).dataframe[inline_column].tolist() == ["old"]


@pytest.mark.parametrize("mode", ["create", "adopt"])
def test_initial_publication_syncs_csv_contents_and_names_before_the_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: Literal["create", "adopt"]
) -> None:
    core = tmp_path / "trajectory.csv"
    metadata = tmp_path / "metadata.csv"
    if mode == "adopt":
        core.write_text("step,action,reward,observation\n1,{},1.0,[]\n")
        metadata.write_text("step\n1\n")
    fsync, replace = os.fsync, os.replace
    synced = set()
    checked = False

    def observe_sync(descriptor: int) -> None:
        fsync(descriptor)
        descriptor_stat = os.fstat(descriptor)
        identity = descriptor_stat.st_dev, descriptor_stat.st_ino
        if identity == (tmp_path.stat().st_dev, tmp_path.stat().st_ino):
            if {"trajectory.csv", "metadata.csv"} <= synced:
                synced.add("directory-after-files")
        else:
            for path in (core, metadata):
                if path.exists() and identity == (path.stat().st_dev, path.stat().st_ino):
                    synced.add(path.name)

    def observe_marker(source: str | Path, destination: str | Path) -> None:
        nonlocal checked
        if Path(destination).name == ".trajectory.commit.json" and core.exists() and metadata.exists():
            checked = True
            assert {"trajectory.csv", "metadata.csv", "directory-after-files"} <= synced
        replace(source, destination)

    monkeypatch.setattr(os, "fsync", observe_sync)
    monkeypatch.setattr(os, "replace", observe_marker)
    owner = Trajectory(iteration_dir=tmp_path, mode=mode)
    if mode == "create":
        owner.append(step=1, action={}, reward=1.0, observation=[])
    assert checked
    assert [record.step for record in Trajectory.load(tmp_path).records()] == [1]


def test_initialization_retry_syncs_the_ancestor_left_by_a_failed_directory_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    iteration = tmp_path / "new-parent" / "iteration"
    ancestor = tmp_path.stat().st_dev, tmp_path.stat().st_ino
    fsync = os.fsync
    failed = False
    retry_synced_ancestor = False

    def fail_ancestor_sync(descriptor: int) -> None:
        nonlocal failed
        descriptor_stat = os.fstat(descriptor)
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) == ancestor and not failed:
            failed = True
            raise OSError("injected ancestor entry sync failure")
        fsync(descriptor)

    with monkeypatch.context() as fault:
        fault.setattr(os, "fsync", fail_ancestor_sync)
        with pytest.raises(OSError, match="injected ancestor entry sync failure"):
            Trajectory(iteration_dir=iteration)
    assert failed

    def observe_retry_sync(descriptor: int) -> None:
        nonlocal retry_synced_ancestor
        fsync(descriptor)
        descriptor_stat = os.fstat(descriptor)
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) == ancestor:
            retry_synced_ancestor = True

    monkeypatch.setattr(os, "fsync", observe_retry_sync)
    owner = Trajectory(iteration_dir=iteration)
    owner.append(step=1, action={}, reward=1.0, observation=[])
    assert retry_synced_ancestor
    assert [record.step for record in Trajectory.load(iteration).records()] == [1]


@pytest.mark.parametrize("mode", ["create", "resume", "adopt"])
@pytest.mark.parametrize("link_kind", ["symlink", "hardlink", "directory"])
@pytest.mark.parametrize("artifact", ["trajectory.csv", "metadata.csv"])
def test_writer_requires_an_independent_regular_csv_pair(
    tmp_path: Path, mode: Literal["create", "resume", "adopt"], link_kind: str, artifact: str
) -> None:
    history = tmp_path / "history"
    shared = tmp_path / "shared"
    history.mkdir()
    shared.mkdir()
    core = history / "trajectory.csv"
    metadata = history / "metadata.csv"
    marker = history / ".trajectory.commit.json"
    if mode == "resume":
        owner = Trajectory(iteration_dir=history)
        owner.append(step=1, action={"x": 1}, reward=1.0, observation=[])
        del owner
        gc.collect()
    elif mode == "adopt":
        core.write_text("step,action,reward,observation\n1,{'x': 1},1.0,[]\n")
        metadata.write_text("step\n1\n")
    else:
        core.write_bytes(b"")
        metadata.write_bytes(b"")
    selected = history / artifact
    external = shared / artifact
    selected.rename(external)
    if link_kind == "symlink":
        selected.symlink_to(external)
    elif link_kind == "hardlink":
        selected.hardlink_to(external)
    else:
        selected.mkdir()
    paths = (core, metadata, marker, external)
    before = {path: path.read_bytes() if path.is_file() else None for path in paths}
    if mode != "create" and link_kind != "directory":
        assert [record.step for record in Trajectory.load(history).records()] == [1]

    with pytest.raises(configurator.TrajectoryLoadError, match=artifact):
        Trajectory(iteration_dir=history, mode=mode)

    assert {path: path.read_bytes() if path.is_file() else None for path in paths} == before
    assert selected.is_symlink() == (link_kind == "symlink")
    assert selected.is_dir() == (link_kind == "directory")


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_late_metadata_alias_refusal_preserves_the_unpublished_core_tail(tmp_path: Path, link_kind: str) -> None:
    history = tmp_path / "history"
    owner = Trajectory(iteration_dir=history)
    owner.append(step=1, action={"x": 1}, reward=1.0, observation=[], env_params={"label": "first"})
    with owner.output_path.open("a", newline="") as stream:
        csv.writer(stream).writerow([2, {"x": 2}, 2.0, []])
    external = tmp_path / "shared-metadata.csv"
    owner.metadata_path.rename(external)
    if link_kind == "symlink":
        owner.metadata_path.symlink_to(external)
    else:
        owner.metadata_path.hardlink_to(external)
    paths = (owner.output_path, owner.metadata_path, history / ".trajectory.commit.json", external)
    before = {path: path.read_bytes() for path in paths}

    with pytest.raises(configurator.TrajectoryLoadError, match=r"metadata\.csv"):
        owner.append(step=2, action={"x": 2}, reward=2.0, observation=[], env_params={"label": "second"})

    assert {path: path.read_bytes() for path in paths} == before
    assert [record.step for record in Trajectory.load(history).records()] == [1]


@pytest.mark.parametrize("partial_tail", [False, True], ids=["complete-unpublished-row", "partial-unpublished-row"])
def test_marker_appearing_after_legacy_precheck_bounds_the_published_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, partial_tail: bool
) -> None:
    core = tmp_path / Trajectory.file_name
    core.write_text("step,action,reward,observation\n1,{'x': 1},10,[1.0]\n")
    marker = tmp_path / ".trajectory.commit.json"
    open_path, replace = Path.open, os.replace
    owners = []
    started = False

    def interrupt_publication(source, target):
        if Path(target) == marker and json.loads(Path(source).read_text())["generation"] == 2:
            raise OSError("unpublished second record")
        return replace(source, target)

    def start_adoption_at_read(path: Path, *args, **kwargs):
        nonlocal started
        if path == core and not started:
            started = True
            owner = Trajectory(iteration_dir=tmp_path, mode="adopt")
            owners.append(owner)
            with pytest.raises(OSError, match="unpublished second record"):
                owner.append(step=2, action={"x": 2}, reward=20, observation=[2.0])
            if partial_tail:
                content = core.read_bytes()
                core.write_bytes(content[: content.rfind(b"\n", 0, len(content) - 1) + 1] + b'2,"unfinished')
        return open_path(path, *args, **kwargs)

    with monkeypatch.context() as scheduling:
        scheduling.setattr(Path, "open", start_adoption_at_read)
        scheduling.setattr(os, "replace", interrupt_publication)
        loaded = Trajectory.load(tmp_path)

    assert started
    assert [record.step for record in loaded.records()] == [1]
    assert json.loads(marker.read_text())["last_step"] == 1
    assert [record.step for record in Trajectory.load(tmp_path).records()] == [1]


@pytest.mark.parametrize("label", ["plain", "bare\rreturn", "line\nfeed", "paired\r\nbreak"])
def test_csv_control_characters_roundtrip_load_resume_and_exact_retry(tmp_path: Path, label: str) -> None:
    writer = Trajectory(iteration_dir=tmp_path)
    values: dict[str, Any] = {
        "step": 1,
        "action": {"x": 1},
        "reward": 10,
        "observation": [1.0],
        "logging": {"label": label},
    }
    writer.append(**values)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    assert Trajectory.load(tmp_path).records()[0]["logging"] == {"label": label}
    del writer
    gc.collect()

    resumed = Trajectory(iteration_dir=tmp_path, mode="resume")
    resumed.append(**values)

    assert Trajectory.load(tmp_path).records()[0]["logging"] == {"label": label}
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before


def test_old_python310_adopted_quoted_carriage_return_keeps_valid_marker_and_retry_bytes(tmp_path: Path) -> None:
    import shutil

    reference = Path(__file__).parent / "ref_data" / "trajectory" / "old310-adopt-quoted-cr"
    expected = {path.name: path.read_bytes() for path in reference.iterdir()}
    shutil.copytree(reference, tmp_path, dirs_exist_ok=True)

    loaded = Trajectory.load(tmp_path)

    assert [record.step for record in loaded.records()] == [1]
    assert loaded.records()[0]["note"] == "left\rright"
    resumed = Trajectory(iteration_dir=tmp_path, mode="resume")
    resumed.append(step=1, action={}, reward=1, observation=[1.0], note="left\rright")
    assert {name: (tmp_path / name).read_bytes() for name in expected} == expected
    assert {path.name: path.read_bytes() for path in reference.iterdir()} == expected
