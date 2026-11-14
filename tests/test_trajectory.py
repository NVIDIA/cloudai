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

from cloudai.configurator import Trajectory, TrajectoryLoadError, TrajectoryRecord
from cloudai.core import CmdArgs, RewardOverrides, TestRun


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

    trajectory.append(step=2, action={"x": 2}, reward=2.0, observation={"metric": 2.0})
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

    with pytest.raises(TrajectoryLoadError, match="missing required columns"):
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
    assert Trajectory.load(tmp_path).records()[0].metadata["logging"] == {"power": "600.0"}


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
    import random

    trajectory = Trajectory(iteration_dir=tmp_path)
    random.seed(0)
    recorded = []
    for step in range(1, 201):
        action = {"x": random.randint(0, 9), "y": random.choice(["a", "b"])}
        env = {"regime": random.choice([0.0, 0.5])}
        trajectory.append(step=step, action=action, reward=step / 100, observation=[float(step)], env_params=env)
        recorded.append((action, env))

    for action, env in recorded:
        expected = next(
            step
            for step, (recorded_action, recorded_env) in enumerate(recorded, start=1)
            if recorded_action == action and recorded_env == env
        )
        found = trajectory.find(action=action, env_params=env)

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
    assert records[0].metadata["execution"] == {"ticket": "T-1"}
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
    assert loaded.records()[0].metadata["source_row"] == {"label": "001"}


def test_loaded_views_and_records_do_not_share_mutable_state(tmp_path: Path) -> None:
    written = Trajectory(iteration_dir=tmp_path)
    written.append(step=1, action={"values": [1, 2]}, reward=1.0, observation=[2.0], execution={"ticket": "T-1"})

    loaded = Trajectory.load(tmp_path)
    record = loaded.records()[0]

    record["action"]["values"].append(3)
    record.metadata["execution"]["ticket"] = "changed"

    flat = loaded.dataframe
    values: object = flat.at[0, "action.values"]
    assert isinstance(values, list)
    values.append(3)

    serialized = loaded.dataframe
    serialized.loc[0, "reward"] = 9.0

    assert loaded.records()[0]["action"] == {"values": [1, 2]}
    assert loaded.records()[0].metadata["execution"] == {"ticket": "T-1"}

    assert loaded.dataframe.loc[0, "action.values"] == [1, 2]
    assert loaded.dataframe.loc[0, "reward"] == 1.0


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


@pytest.mark.parametrize("domain", ["step", "action", "reward", "observation"])
def test_inline_auxiliary_columns_cannot_change_core_values(tmp_path: Path, domain: str) -> None:
    source = tmp_path / "trajectory.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "action", "reward", "observation", f"{domain}.extra"])
        writer.writerow([1, {}, {}, {}, "auxiliary"])
    original = source.read_bytes()

    with pytest.raises(TrajectoryLoadError, match="overlaps core domain") as failure:
        Trajectory.load(source)

    assert str(source) in str(failure.value)
    assert domain in str(failure.value)
    assert source.read_bytes() == original


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
    assert record.metadata == {}
    assert written.metadata_path.read_text().splitlines() == ["step", "1"]
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "trajectory.csv").write_bytes(written.output_path.read_bytes())
    assert dict(Trajectory.load(legacy).records()[0]) == dict(record)
    assert not (legacy / "metadata.csv").exists()


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
    ],
)
def test_load_rejects_inconsistent_artifacts_before_returning_history(
    tmp_path: Path, core: str, metadata: str | None, artifact: str
) -> None:
    (tmp_path / "trajectory.csv").write_text(core)
    if metadata is not None:
        (tmp_path / "metadata.csv").write_text(metadata)
    original_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    with pytest.raises(TrajectoryLoadError, match=artifact) as failure:
        Trajectory.load(tmp_path)

    assert str(failure.value).startswith(f"Cannot load trajectory {tmp_path / artifact}:")
    assert "row " in str(failure.value) or "step " in str(failure.value)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original_files


@pytest.mark.parametrize("artifact", ["trajectory.csv", "metadata.csv"])
def test_load_reports_unreadable_or_undecodable_artifact(tmp_path: Path, artifact: str) -> None:
    (tmp_path / "trajectory.csv").write_text("step,action,reward,observation\n1,{'ooo': 10},0.1,[10.0]\n")
    (tmp_path / artifact).write_bytes(b"\xff\xfe")

    with pytest.raises(TrajectoryLoadError, match=artifact):
        Trajectory.load(tmp_path)


def test_load_reports_missing_required_artifact(tmp_path: Path) -> None:
    with pytest.raises(TrajectoryLoadError, match=r"trajectory\.csv"):
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

    with pytest.raises(TrajectoryLoadError) as failure:
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

    with pytest.raises(TrajectoryLoadError) as failure:
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
        assert loaded.records()[0].metadata["logging"] == {"payload": payload}
        assert loaded.dataframe["step"].tolist() == [record.step for record in loaded.records()]
        assert csv.field_size_limit() == 4096

        written.metadata_path.write_text(f'step,logging.payload\n1,"{payload}",extra\n')

        with pytest.raises(TrajectoryLoadError):
            Trajectory.load(tmp_path)

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

    with pytest.raises(TrajectoryLoadError):
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


@pytest.mark.parametrize(
    ("observation", "present", "expected"),
    [
        (None, False, "n/a"),
        ([], True, "n/a"),
        ("[truncated observation]", True, "n/a"),
        ([1.23456, 2.0, "ready", 0.00001], True, "1.2346, 2, ready, 0"),
        (0, True, "0"),
    ],
)
def test_record_formats_observation_without_changing_its_domain(
    tmp_path: Path, observation: Any, present: bool, expected: str
) -> None:
    if present:
        trajectory = Trajectory(iteration_dir=tmp_path)
        trajectory.append(step=1, action={"ooo": 10}, reward=1.0, observation=observation)
    else:
        trajectory = Trajectory(
            iteration_dir=tmp_path,
            dataframe=pd.DataFrame([{"step": 1, "action.ooo": 10, "reward": 1.0}]),
        )
    record = trajectory.records()[0]
    original_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    before = repr(record["observation"]) if present else None

    assert record.format_observation() == expected

    if present:
        assert repr(record["observation"]) == before
    else:
        with pytest.raises(KeyError):
            _ = record["observation"]

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original_files


@pytest.fixture(params=[False, True], ids=["live", "loaded"])
def readonly_record(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[TrajectoryRecord, MappingProxyType[str, Any]]:
    expected = {
        "step": 1,
        "action": {"values": ["1", "2.5", "auto"], "label": "001"},
        "reward": 0.5,
        "observation": [2.0],
        "env_params": {"message_size": "1024"},
        "execution": {"ticket": "T-1"},
        "source": {"archive": "prior"},
    }
    trajectory = Trajectory(iteration_dir=tmp_path)
    trajectory.append(**expected)
    if request.param:
        trajectory = Trajectory.load(tmp_path)

    return trajectory.records()[0], MappingProxyType(expected)


def test_record_reads_match_a_readonly_dictionary(
    readonly_record: tuple[TrajectoryRecord, MappingProxyType[str, Any]], tmp_path: Path
) -> None:
    record, expected = readonly_record

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
    readonly_record: tuple[TrajectoryRecord, MappingProxyType[str, Any]],
) -> None:
    record, expected = readonly_record
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


def test_resolved_context_round_trips_types_literal_paths_and_raw_metadata(
    base_tr: TestRun, tmp_path: Path, slurm_system
) -> None:
    payload = {
        "numeric_string": "001",
        "bool_string": "true",
        "null_string": "null",
        "list_string": "[]",
        "number": 1,
        "boolean": True,
        "null": None,
        "empty_mapping": {},
        "empty_list": [],
        "mixed": ["001", 1, True, None, {}],
        "nested": {"a.b": "literal", "a": {"b": "nested"}},
    }
    base_tr.test.cmd_args = CmdArgs.model_validate({"tuned": [1, 2], "payload": payload})
    base_tr.test.dse_excluded_args = ["cmd_args.payload"]
    action_space = base_tr.param_space
    resolved = base_tr.apply_params_set({"tuned": 2})
    resolved.step = 1
    written = Trajectory(iteration_dir=tmp_path)

    written.record(
        test_run=resolved,
        action_space=action_space,
        action={"tuned": 2},
        reward=1.0,
        observation=[2.0],
        system=slurm_system,
        rewards=RewardOverrides(),
    )

    for record in (written.records()[0], Trajectory.load(tmp_path).records()[0]):
        context = record.get("env_params", decode_numbers=True)
        assert context["payload"] == payload
        assert {key: type(value) for key, value in context["payload"].items()} == {
            key: type(value) for key, value in payload.items()
        }
        assert [type(value) for value in context["payload"]["mixed"]] == [str, int, bool, type(None), dict]
        assert "tuned" not in context
        raw = record["env_params"]["payload"]
        assert raw["numeric_string"] == '"001"'
        assert raw["boolean"] == "true" and raw["null"] == "null"
        assert raw["empty_mapping"] == "{}" and raw["empty_list"] == "[]"
        assert raw["nested"] == {"a.b": '"literal"', "a": {"b": '"nested"'}}
        assert record.metadata["env_params"] == record["env_params"] == record.get("env_params")
        context["payload"]["mixed"].append("changed")
        raw["nested"]["a.b"] = "changed"
        assert record.get("env_params", decode_numbers=True)["payload"] == payload
        assert record["env_params"]["payload"]["nested"]["a.b"] == '"literal"'


@pytest.mark.parametrize(
    "version, fields, recovery",
    [
        ("99", {'env_params.["value"]': "1"}, None),
        ("", {'env_params.["value"]': "1"}, None),
        ("1", {"env_params.not-an-array": "1"}, None),
        ("1", {"env_params.[]": "1"}, None),
        ("1", {"env_params.[1]": "1"}, None),
        ("1", {'env_params.["value"]': "not-json"}, None),
        ("1", {'env_params.["value"]': "NaN"}, None),
        ("1", {'env_params.["value"]': "Infinity"}, None),
        ("1", {'env_params.["value"]': "1e999"}, None),
        ("1", {'env_params.["value"]': '{"a":1,"a":2}'}, None),
        ("1", {'env_params.["value"]': "1", "execution..ticket": "invalid"}, None),
        ("1", {'env_params.["value"]': "1", 'env_params.[ "value" ]': "2"}, None),
        ("1", {'env_params.["value"]': "1", 'env_params.["value","child"]': "2"}, None),
        (None, {"record_format_version.part": "1", "env_params.literal": "001"}, None),
        (None, {"record_format_version.part": "1", "env_params.literal": "001"}, "recorded"),
    ],
)
def test_loading_rejects_unknown_or_ambiguous_context_encoding(
    tmp_path: Path, version: str | None, fields: dict[str, str], recovery: Literal["recorded"] | None
) -> None:
    (tmp_path / "trajectory.csv").write_text("step,action,reward,observation\n1,{},1.0,[]\n")
    controls = {"record_format_version": version} if version is not None else {}
    with (tmp_path / "metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", *controls, *fields])
        writer.writeheader()
        writer.writerow({"step": 1, **controls, **fields})
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(TrajectoryLoadError):
        Trajectory.load(tmp_path, recovery=recovery)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("unsupported", [float("nan"), float("inf"), object()])
def test_invalid_resolved_context_changes_neither_csv(
    base_tr: TestRun, tmp_path: Path, unsupported: object, slurm_system
) -> None:
    base_tr.test.cmd_args = CmdArgs.model_validate({"payload": unsupported})
    base_tr.step = 1
    trajectory = Trajectory(iteration_dir=tmp_path)

    with pytest.raises((TypeError, ValueError)):
        trajectory.record(
            test_run=base_tr,
            action_space={},
            action={},
            reward=1.0,
            observation=[],
            system=slurm_system,
            rewards=RewardOverrides(),
        )

    assert not trajectory.output_path.exists()
    assert not trajectory.metadata_path.exists()
    assert len(trajectory) == 0


@pytest.mark.parametrize("reserved", ["extra_env_vars", "extra_cmd_args", "NUM_NODES", "test_definition"])
def test_resolved_context_refuses_reserved_command_names_before_publication(
    base_tr: TestRun, tmp_path: Path, reserved: str, slurm_system
) -> None:
    base_tr.test.cmd_args = CmdArgs(**{reserved: "workload-value"})
    base_tr.step = 1
    trajectory = Trajectory(iteration_dir=tmp_path)

    with pytest.raises(ValueError):
        trajectory.record(
            test_run=base_tr,
            action_space={},
            action={},
            reward=1.0,
            observation=[],
            system=slurm_system,
            rewards=RewardOverrides(),
        )

    assert not trajectory.output_path.exists()
    assert not trajectory.metadata_path.exists()


@pytest.mark.parametrize("metadata", [{"parent": "value", "parent.child": "conflict"}, {"parent..child": "empty"}])
def test_complete_record_validation_precedes_both_csv_writes(tmp_path: Path, metadata: dict[str, str]) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.suffix == ".csv"}

    with pytest.raises((ValueError, TrajectoryLoadError)):
        trajectory.append(step=1, action={"x": 1}, reward=1.0, observation=[], details=metadata)

    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.suffix == ".csv"}
    assert len(trajectory) == 0


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
    tmp_path: Path, slurm_system, with_metadata: bool, reordered: bool
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
    before_recovery = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    recovered = Trajectory.load(tmp_path, recovery="recorded").records()
    assert [record.step for record in recovered] == [0, 3, 4]
    assert all(
        record.get("run_details", decode_numbers=True)["recovery"]["source_id_kind"] == "history_uuid"
        for record in recovered
    )
    marker_identity = json.loads((tmp_path / ".trajectory.commit.json").read_text())["history_id"]
    assert all(
        record.get("run_details", decode_numbers=True)["recovery"]["source_id"] == marker_identity
        for record in recovered
    )
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before_recovery
    assert all((tmp_path / name).read_bytes().startswith(content) for name, content in before.items())
    from cloudai.core import TestDefinition

    run = TestRun(
        name="current",
        test=TestDefinition(name="current", description="current", test_template_name="current", cmd_args=CmdArgs()),
        num_nodes=1,
        nodes=[],
        step=5,
    )
    with pytest.raises((ValueError, TrajectoryLoadError)):
        resumed.record(
            test_run=run,
            system=slurm_system,
            rewards=RewardOverrides(),
            action_space={},
            action={},
            reward=4.0,
            observation=[],
        )
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before_recovery


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

    with pytest.raises(TrajectoryLoadError):
        Trajectory.load(tmp_path)
    with pytest.raises((ValueError, TrajectoryLoadError)):
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

    with pytest.raises(TrajectoryLoadError):
        Trajectory.load(tmp_path)


@pytest.mark.parametrize("contents", ["{", "{}"])
def test_malformed_commit_marker_is_an_error_even_when_csvs_are_valid(tmp_path: Path, contents: str) -> None:
    owner = Trajectory(iteration_dir=tmp_path)
    owner.append(step=1, action={}, reward=1.0, observation=[])
    (tmp_path / ".trajectory.commit.json").write_text(contents)

    with pytest.raises(TrajectoryLoadError):
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

    with pytest.raises(TrajectoryLoadError):
        Trajectory(iteration_dir=tmp_path, mode="adopt")

    assert before == {path.name: path.read_bytes() for path in tmp_path.glob("*.csv")}
    assert not (tmp_path / ".trajectory.commit.json").exists()


@pytest.mark.parametrize("legacy_message, first_matching_step", [(10, 2), (20, 1)])
def test_warm_legacy_cache_and_versioned_records_share_earliest_exact_lookup(
    tmp_path: Path, legacy_message: int, first_matching_step: int
) -> None:
    dataframe = pd.DataFrame(
        [{"step": 1, "action.x": 1, "reward": 1.0, "observation.metric": 1.0, "env_params.message": legacy_message}],
        dtype=object,
    )
    trajectory = Trajectory(iteration_dir=tmp_path, dataframe=dataframe)
    trajectory.append(
        step=2,
        action={"x": 1},
        reward=2.0,
        observation={"metric": 2.0},
        env_params={"message": 20},
        record_format_version="1",
    )

    found = trajectory.find(action={"x": 1}, env_params={"message": 20})
    assert found is not None and found["step"] == first_matching_step
    loaded = Trajectory.load(tmp_path)
    found = loaded.find(action={"x": 1}, env_params={"message": 20})
    assert found is not None and found["step"] == 2
    assert [record.step for record in loaded.records()] == [2]


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
        with pytest.raises(TrajectoryLoadError):
            Trajectory.load(source)
        loaded = Trajectory.load(source, missing_ok=True)
        assert loaded.records() == ()
        assert loaded.dataframe.empty
        with pytest.raises(RuntimeError, match="read-only"):
            loaded.append(step=1, action={}, reward=1.0, observation=[])
    else:
        with pytest.raises(TrajectoryLoadError):
            Trajectory.load(source, missing_ok=True)
    assert sorted(path.name for path in tmp_path.iterdir()) == before


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

    with pytest.raises(TrajectoryLoadError, match=artifact):
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

    with pytest.raises(TrajectoryLoadError, match=r"metadata\.csv"):
        owner.append(step=2, action={"x": 2}, reward=2.0, observation=[], env_params={"label": "second"})

    assert {path: path.read_bytes() for path in paths} == before
    assert [record.step for record in Trajectory.load(history).records()] == [1]


@pytest.mark.parametrize(
    "reward,observation", [(float("nan"), [1.0]), (1.0, [float("inf")]), (float("-inf"), [float("nan"), float("-inf")])]
)
def test_record_preserves_nonfinite_measurements_outside_the_strict_context_codec(
    tmp_path: Path, reward, observation
) -> None:
    from cloudai.core import TestDefinition
    from cloudai.systems.standalone import StandaloneSystem

    test = TestDefinition(
        name="nonfinite",
        description="nonfinite",
        test_template_name="nonfinite",
        cmd_args=CmdArgs.model_validate({"choice": 2, "fixed": "001"}),
    )
    run = TestRun(name="nonfinite", test=test, num_nodes=1, nodes=[], step=1)
    system = StandaloneSystem(name="nonfinite", install_path=tmp_path / "install", output_path=tmp_path / "output")
    writer = Trajectory(iteration_dir=tmp_path / "history")
    record = writer.record(
        test_run=run,
        system=system,
        action_space={"choice": [1, 2]},
        action={"choice": 2},
        reward=reward,
        observation=observation,
        rewards=RewardOverrides(),
    )
    assert isinstance(record, TrajectoryRecord)
    reloaded = Trajectory.load(writer.output_path).records()[0]
    assert math.isnan(reloaded["reward"]) if math.isnan(reward) else reloaded["reward"] == reward
    assert all(
        math.isnan(actual) if math.isnan(expected) else actual == expected
        for actual, expected in zip(reloaded["observation"], observation, strict=True)
    )
    assert reloaded.get("env_params", decode_numbers=True)["fixed"] == "001"


@pytest.mark.parametrize(
    ("control", "version"),
    [("record_context_version", "999"), ("record_context_version", 1), ("record_format_version.part", "1")],
)
def test_unknown_context_codec_is_refused_before_either_csv_is_written(
    tmp_path: Path, control: str, version: str | int
) -> None:
    writer = Trajectory(iteration_dir=tmp_path)
    values: dict[str, object] = {control: version}
    if control == "record_context_version":
        values.update({"run_details": "{}", "measurement": "{}", "evaluation_failure": "null"})
    else:
        values["env_params"] = {"literal": "001"}
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises((ValueError, TrajectoryLoadError)):
        writer.append(step=1, action={"x": 1}, reward=1.0, observation=[], **values)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("domain", ["test_run", "action_space", "system", "rewards", "context"])
def test_generic_find_retains_metadata_names_used_by_recording_context(tmp_path: Path, domain: str) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)
    ordinary = {"kind": "ordinary-metadata"}
    trajectory.append(step=1, action={}, reward=1.0, observation=[], **{domain: ordinary})

    found = trajectory.find(**{domain: ordinary})

    assert found is not None
    assert found["step"] == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (None, None),
        ("record_context_version", "999"),
        ("evaluation_failure", None),
        ("measurement", "{"),
        ("run_details", '{"a":1,"a":2}'),
        ("measurement", '{"a":NaN}'),
        ("run_details", "[]"),
        ("record_context_version.extra", "1"),
        ("run_details.extra", "{}"),
    ],
)
def test_declared_context_cells_preserve_raw_views_and_refuse_malformed_archives(
    tmp_path: Path, slurm_system, field: str | None, replacement: str | None
) -> None:
    from cloudai.core import TestDefinition

    test = TestDefinition(
        name="codec", description="codec", test_template_name="codec", cmd_args=CmdArgs.model_validate({"fixed": "001"})
    )
    run = TestRun(name="codec", test=test, num_nodes=1, nodes=[], step=1)
    writer = Trajectory(iteration_dir=tmp_path / "source")
    original = writer.record(
        test_run=run,
        system=slurm_system,
        action_space={},
        action={},
        reward=1.0,
        observation=[],
        rewards=RewardOverrides(),
    )
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "trajectory.csv").write_bytes(writer.output_path.read_bytes())
    sidecar = archive / "metadata.csv"
    with writer.metadata_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    fields.append("source_row")
    rows[0]["source_row"] = "001"
    if field is not None:
        if replacement is None:
            fields.remove(field)
            rows[0].pop(field)
        else:
            if field not in fields:
                fields.append(field)
            rows[0][field] = replacement
    with sidecar.open("w", newline="") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    before = {path.name: path.read_bytes() for path in archive.iterdir()}

    if field is None:
        record = Trajectory.load(archive).records()[0]
        assert isinstance(record.get("run_details", decode_numbers=True), dict)
        assert isinstance(record.get("measurement", decode_numbers=True), dict)
        assert record.get("evaluation_failure", decode_numbers=True) is None
        for domain in ("run_details", "measurement", "evaluation_failure"):
            assert isinstance(record[domain], str)
            assert record.get(domain) == record.metadata[domain] == record[domain]
            assert record.get(domain, decode_numbers=True) == original.get(domain, decode_numbers=True)
        assert record.get("source_row", decode_numbers=True) == "001"
    else:
        with pytest.raises(TrajectoryLoadError):
            Trajectory.load(archive)
    assert {path.name: path.read_bytes() for path in archive.iterdir()} == before


@pytest.fixture
def legacy_recorded_history(tmp_path: Path) -> Path:
    import toml

    from cloudai.core import CommandGenStrategy

    root = tmp_path / "history"
    root.mkdir()
    with (root / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "action", "reward", "observation"])
        writer.writerows((step, {"x": 1}, 2.0, [0.003]) for step in (0, 2, 7))
    with (root / "metadata.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "env_params.label",
                "env_params.enabled",
                "env_params.items",
                "env_params.empty",
                "run_details.label",
                "source_row",
                "legacy_json",
            ]
        )
        writer.writerows((step, "001", True, ["001", "true"], [], "001", "001", '{"x":1}') for step in (0, 2, 7))
    recorded = {
        "name": "historical-name",
        "current_iteration": 9,
        "step": 0,
        "output_path": "/historical/path/that/is/not/a/lookup/location",
        "nnodes": 2,
        "test_definition": {
            "cmd_args": {
                "x": 1,
                "label": "001",
                "enabled": True,
                "items": ["001", "true"],
                "empty": [],
                "nested.value": "literal",
                "nested": {"value": "inside", "empty": {}},
                "redacted_label": "<REDACTED>",
                "drop_rate": 0.0,
            },
            "agent_metrics": ["bus_bw"],
            "agent_observation": ["drop_rate"],
            "agent_reward_function": "identity",
            "agent_config": {"rewards": {"metric_failure": -7}},
        },
    }
    (root / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).write_text(toml.dumps(recorded))
    unverified = root / "7"
    unverified.mkdir()
    (unverified / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).write_text(
        toml.dumps({"name": "unverified", "test_definition": {"cmd_args": {"label": "not-trusted"}}})
    )
    return root


@pytest.mark.parametrize(
    ("history", "dump_state"),
    [
        ("legacy", "matched"),
        ("legacy", "absent"),
        ("legacy", "malformed"),
        ("modern", "matched"),
        ("modern", "absent"),
        ("modern", "malformed"),
        ("typed-environment", "matched"),
        ("scalar-action", "absent"),
    ],
)
def test_recorded_recovery_preserves_default_and_modern_histories(
    tmp_path: Path, slurm_system, history: str, dump_state: str
) -> None:
    import toml

    from cloudai.core import CommandGenStrategy, TestDefinition

    root = tmp_path / "history"
    root.mkdir()
    action = 42 if history == "scalar-action" else {"x": 1}
    if history == "modern":
        definition = TestDefinition(
            name="modern", description="modern", test_template_name="modern", cmd_args=CmdArgs.model_validate({"x": 1})
        )
        run = TestRun(name="modern", test=definition, num_nodes=1, nodes=[], step=1)
        writer = Trajectory(iteration_dir=root)
        writer.record(
            test_run=run,
            system=slurm_system,
            rewards=RewardOverrides(),
            action_space={"x": [1, 2]},
            action=action,
            reward=1.0,
            observation=[],
        )
    elif history in {"typed-environment", "scalar-action"}:
        writer = Trajectory(iteration_dir=root)
        metadata = (
            {"record_format_version": "1", "env_params": {"fixed": True}} if history == "typed-environment" else {}
        )
        writer.append(step=1, action=action, reward=1.0, observation=[], **metadata)
    else:
        (root / "trajectory.csv").write_text("step,action,reward,observation\n1,{'x': 1},1.0,[]\n")
    dump = root / "1" / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME
    if dump_state != "absent":
        dump.parent.mkdir(exist_ok=True)
        recorded_args = {"x": 1, "fixed": True} if history == "typed-environment" else {"x": 1}
        dump.write_text(
            "[malformed"
            if dump_state == "malformed"
            else toml.dumps({"step": 1, "test_definition": {"cmd_args": recorded_args}})
        )
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    raw = Trajectory.load(root).records()[0]
    assert raw["action"] == action
    assert raw["reward"] == 1.0 and raw["observation"] == []
    if history == "typed-environment":
        assert raw.get("env_params", decode_numbers=True)["fixed"] is True
    if history == "modern":
        recovered = Trajectory.load(root, recovery="recorded").records()[0]
        assert dict(recovered) == dict(raw) and recovered.metadata == raw.metadata
        assert "recovery" not in recovered.get("run_details", decode_numbers=True)
    elif dump_state == "malformed":
        with pytest.raises(TrajectoryLoadError):
            Trajectory.load(root, recovery="recorded")
    else:
        recovered = Trajectory.load(root, recovery="recorded").records()[0]
        expected_environment = {"fixed": True} if history == "typed-environment" else {}
        assert recovered.get("env_params", decode_numbers=True) == expected_environment
        assert (recovered["action"], recovered["reward"], recovered["observation"]) == (
            raw["action"],
            raw["reward"],
            raw["observation"],
        )
        details = recovered.get("run_details", decode_numbers=True)
        if not expected_environment:
            assert isinstance(details["recovery"]["unknowns"]["env_params"], str)
        else:
            assert recovered.get("env_params", decode_numbers=True)["fixed"] is True
        assert type(recovered["action"]) is type(raw["action"])
        assert recovered.get("configuration_id") is None
    assert {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_recorded_recovery_uses_only_matching_historical_evidence(
    legacy_recorded_history: Path, tmp_path: Path
) -> None:
    import shutil

    from cloudai.core import CommandGenStrategy

    root = legacy_recorded_history
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    raw = Trajectory.load(root).records()
    records = Trajectory.load(root, recovery="recorded").records()
    assert [record.step for record in records] == [0, 2, 7]
    assert [(record["action"], record["reward"], record["observation"]) for record in records] == [
        (record["action"], record["reward"], record["observation"]) for record in raw
    ]
    assert records[0].get("env_params", decode_numbers=True) == {
        "label": "001",
        "enabled": True,
        "items": ["001", "true"],
        "empty": [],
        "nested.value": "literal",
        "nested": {"value": "inside", "empty": {}},
        "redacted_label": "<REDACTED>",
        "NUM_NODES": 2,
        "drop_rate": 0.0,
    }
    details = records[0].get("run_details", decode_numbers=True)
    assert details["agent_metrics"] == ["bus_bw"] and details["agent_observation"] == ["drop_rate"]
    assert details["reward_function"] == "identity"
    assert details["launch"]["num_nodes"] == 2 and "nodes" not in details["launch"]
    assert "evaluation_policy" not in details
    recovery = details["recovery"]
    assert recovery["policy"] == "recorded" and recovery["source_id_kind"] == "sha256"
    assert recovery["historical_identity"] == {
        "name": "historical-name",
        "current_iteration": 9,
        "step": 0,
        "output_path": "/historical/path/that/is/not/a/lookup/location",
    }
    assert recovery["context_completeness"] == {
        "status": "unknown",
        "reason": "historical_serialization_may_redact_or_omit_fields",
    }
    assert recovery["unknowns"]["action_space"] == "original_declaration_unverified"
    assert recovery["unknowns"]["evaluation_policy"] == "historical_actual_policy_not_recorded"
    assert recovery["ambiguous_env_paths"] == []
    assert all(
        path in recovery["recorded_env_paths"]
        for path in [["label"], ["nested.value"], ["nested", "value"], ["nested", "empty"]]
    )
    assert len(recovery["source_id"]) == 64
    dump_name = CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME
    for record, status, reason, relative in zip(
        records,
        ["matched", "absent", "unverified_identity"],
        [None, "historical_dump_absent", "historical_dump_step_missing"],
        [dump_name, f"2/{dump_name}", f"7/{dump_name}"],
        strict=True,
    ):
        assert record["source_row"] == "001" and record["legacy_json"] == '{"x":1}'
        assert record.get("run_details", decode_numbers=True)["label"] == "001"
        current = record.get("run_details", decode_numbers=True)["recovery"]
        assert current["source_id"] == recovery["source_id"] and current["source_step"] == record.step
        assert current["dump"]["status"] == status and current["dump"]["reason"] == reason
        assert current["dump"]["relative_path"] == relative
        assert current["dump"]["sha256"] == (
            hashlib.sha256((root / relative).read_bytes()).hexdigest() if status != "absent" else None
        )
        measurement = record.get("measurement", decode_numbers=True)
        assert measurement["kind"] == measurement["outcome"] == measurement["execution_status"] == "unknown"
        assert measurement["origin"] == {"status": "unknown", "reason": "historical_measurement_origin_not_recorded"}
        assert record.get("evaluation_failure", decode_numbers=True) == {
            "status": "unknown",
            "reason": "historical_failure_evidence_not_recorded",
        }
        assert record.get("configuration_id") is None
        assert record.source == root / "trajectory.csv"
    for record, original in zip(records[1:], raw[1:], strict=True):
        assert record.get("env_params", decode_numbers=True) == original.get("env_params", decode_numbers=True)
        assert ["label"] in record.get("run_details", decode_numbers=True)["recovery"]["ambiguous_env_paths"]
    copied = tmp_path / "copied"
    shutil.copytree(root, copied)
    relocated = Trajectory.load(copied, recovery="recorded").records()
    assert [record.get("run_details", decode_numbers=True)["recovery"]["source_id"] for record in relocated] == [
        recovery["source_id"]
    ] * 3
    assert all(record.source == copied / "trajectory.csv" for record in relocated)
    assert {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize(
    ("dump_patch", "metadata_patch", "malformed", "expect_conflict"),
    [
        ({"test_definition": {"cmd_args": {"x": 2}}}, {}, False, True),
        ({"test_definition": {"cmd_args": {"label": "different"}}}, {}, False, True),
        (
            {"test_definition": {"agent_metrics": ["different"]}},
            {"run_details.agent_metrics": "['bus_bw']"},
            False,
            True,
        ),
        (
            {"test_definition": {"agent_observation": ["different"]}},
            {"run_details.agent_observation": "['drop_rate']"},
            False,
            True,
        ),
        (
            {"test_definition": {"agent_reward_function": "negative"}},
            {"run_details.reward_function": "identity"},
            False,
            True,
        ),
        ({"step": 8}, {}, False, True),
        ({"step": False}, {}, False, True),
        ({}, {}, True, True),
        ({}, {"run_details": "{'unversioned': 'json-like'}"}, False, True),
        ({}, {"run_details.recovery.source_step": "7"}, False, True),
        ({}, {"agent_metrics": "['different']"}, False, True),
        ({}, {"agent_observation": "['different']"}, False, True),
        ({}, {"agent_reward_function": "negative"}, False, True),
        ({}, {"test_definition.cmd_args.label": "different"}, False, True),
        ({}, {"nnodes": "3"}, False, True),
        ({}, {"name": "different"}, False, True),
        ({}, {"current_iteration": "8"}, False, True),
        (
            {},
            {
                "agent_metrics": "['bus_bw']",
                "agent_observation": "['drop_rate']",
                "agent_reward_function": "identity",
                "test_definition.cmd_args.label": "001",
                "nnodes": "2",
                "name": "historical-name",
                "current_iteration": "9",
            },
            False,
            False,
        ),
    ],
)
def test_recorded_recovery_conflicts_before_import(
    legacy_recorded_history: Path, dump_patch: dict, metadata_patch: dict, malformed: bool, expect_conflict: bool
) -> None:

    import toml

    from cloudai.core import CommandGenStrategy

    root = legacy_recorded_history
    dump_name = CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME
    recorded = toml.loads((root / dump_name).read_text())
    recorded["step"] = 7
    for key, value in dump_patch.items():
        if key == "test_definition":
            for field, child in value.items():
                if field == "cmd_args":
                    recorded[key][field].update(child)
                else:
                    recorded[key][field] = child
        else:
            recorded[key] = value
    selected_dump = root / dump_name if dump_patch.get("step") is False else root / "7" / dump_name
    selected_dump.write_text("[malformed" if malformed else toml.dumps(recorded))
    sidecar = root / "metadata.csv"
    with sidecar.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = list(reader.fieldnames or []), list(reader)
    for name, value in metadata_patch.items():
        if name == "run_details":
            fields = [field for field in fields if not field.startswith("run_details.")]
            rows = [{key: child for key, child in row.items() if key in fields} for row in rows]
        fields.append(name)
        for row in rows:
            row[name] = value
    with sidecar.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    raw = Trajectory.load(root).records()
    assert len(raw) == 3
    if expect_conflict:
        with pytest.raises(TrajectoryLoadError):
            Trajectory.load(root, recovery="recorded")
    else:
        recovered = Trajectory.load(root, recovery="recorded").records()
        assert len(recovered) == 3
        assert [
            {
                name: record[name]
                for name in (
                    "agent_metrics",
                    "agent_observation",
                    "agent_reward_function",
                    "test_definition",
                    "nnodes",
                    "name",
                    "current_iteration",
                )
            }
            for record in recovered
        ] == [
            {
                "agent_metrics": "['bus_bw']",
                "agent_observation": "['drop_rate']",
                "agent_reward_function": "identity",
                "test_definition": {"cmd_args": {"label": "001"}},
                "nnodes": "2",
                "name": "historical-name",
                "current_iteration": "9",
            }
        ] * 3
    assert [record.step for record in raw] == [0, 2, 7]
    assert {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


@pytest.fixture
def eligibility_context(tmp_path: Path):
    from cloudai.configurator.env_params import EnvParamSpec
    from cloudai.core import TestDefinition
    from cloudai.systems.standalone import StandaloneSystem

    test = TestDefinition(
        name="eligibility",
        description="eligibility",
        test_template_name="eligibility",
        cmd_args=CmdArgs.model_validate({"x": [1, 2], "y": [10, 20], "fixed": "001", "drop": [0.0, 0.1]}),
        env_params={"drop": EnvParamSpec(weights=[0.5, 0.5])},
        agent_metrics=["bandwidth"],
        agent_reward_function="identity",
    )
    run = TestRun(name="eligibility", test=test, num_nodes=1, nodes=[], output_path=tmp_path / "run")
    system = StandaloneSystem(name="eligibility", install_path=tmp_path / "install", output_path=tmp_path / "output")
    return run, system, RewardOverrides()


@pytest.mark.parametrize(
    "case,blocked",
    [
        ("minus-one", {}),
        ("negative", {}),
        ("metric-failure", {"measurement": "explicit_failure", "learning": "explicit_failure"}),
        ("runner-failure", {"measurement": "explicit_failure", "learning": "explicit_failure"}),
        (
            "nan-reward",
            {"reference": "reward_unusable", "measurement": "reward_unusable", "learning": "reward_unusable"},
        ),
        ("nonfinite-observation", {"measurement": "observation_unusable", "learning": "observation_unusable"}),
        ("legacy-minus-one", {"measurement": "legacy_failure_ambiguous", "learning": "legacy_failure_ambiguous"}),
        ("legacy-negative", {"learning": "context_unavailable"}),
        (
            "step-zero",
            {purpose: "historical_step_zero" for purpose in ("reference", "measurement", "decision", "learning")},
        ),
        (
            "scalar-action",
            {purpose: "action_unusable" for purpose in ("reference", "measurement", "decision", "learning")},
        ),
        (
            "nonfinite-action",
            {purpose: "action_unusable" for purpose in ("reference", "measurement", "decision", "learning")},
        ),
        ("legacy-empty-failure", {"learning": "context_unavailable"}),
        (
            "malformed-outcome",
            {"measurement": "measurement_schema_incompatible", "learning": "measurement_schema_incompatible"},
        ),
        (
            "malformed-execution-status",
            {"measurement": "measurement_schema_incompatible", "learning": "measurement_schema_incompatible"},
        ),
        (
            "malformed-metric-status",
            {"measurement": "measurement_schema_incompatible", "learning": "measurement_schema_incompatible"},
        ),
        (
            "malformed-native-indices",
            {"measurement": "measurement_schema_incompatible", "learning": "measurement_schema_incompatible"},
        ),
        (
            "legacy-native-status",
            {"measurement": "measurement_schema_incompatible", "learning": "measurement_schema_incompatible"},
        ),
        ("legacy-known-failure", {"measurement": "explicit_failure", "learning": "explicit_failure"}),
    ],
)
def test_record_eligibility_preserves_evidence_for_distinct_consumer_roles(
    tmp_path: Path, eligibility_context, case, blocked
):
    from cloudai.models.output import Metric, Run

    original, system, rewards = eligibility_context
    trajectory: Trajectory | None = None
    invalid_purpose: Any = "unsupported"
    if case.startswith("legacy") or case in ("step-zero", "scalar-action", "nonfinite-action"):
        path = tmp_path / "legacy.csv"
        action = (
            42 if case == "scalar-action" else {"x": float("inf")} if case == "nonfinite-action" else {"x": 1, "y": 10}
        )
        reward = -1.0 if case == "legacy-minus-one" else -2.0
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["step", "action", "reward", "observation"])
            writer.writerow([0 if case == "step-zero" else 7, action, reward, [1.0]])
        opaque_fields = "step,evaluation_failure.metric_indices,evaluation_failure.constraint\n7,[],False\n"
        metadata_fields = {
            "legacy-empty-failure": opaque_fields,
            "legacy-native-status": opaque_fields,
            "legacy-known-failure": "step,measurement.outcome,measurement.metrics.0.status\n7,runner_failure,failed\n",
        }
        if case in metadata_fields:
            (tmp_path / Trajectory.metadata_file_name).write_text(metadata_fields[case])
            expected_failure = ("explicit_failure",) if case == "legacy-known-failure" else ()
            assert Trajectory.load(path).records()[0].ineligibility_reasons("measurement") == expected_failure
        record = Trajectory.load(path, recovery="recorded").records()[0]
    else:
        resolved = original.apply_params_set({"x": 1, "y": 10, "drop": 0.0})
        resolved.step = 7
        reward = float("nan") if case == "nan-reward" else -2.0 if case == "negative" else -1.0
        observation = [float("inf")] if case == "nonfinite-observation" else [1.0]
        output = Run(
            path=str(resolved.output_path),
            jobid="job",
            status="completed",
            metrics=[Metric(name="bandwidth", value=1.0, unit="GB/s")],
        )
        trajectory = Trajectory(iteration_dir=tmp_path / "history")
        record = trajectory.record(
            test_run=resolved,
            action_space=original.param_space,
            action={"x": 1},
            reward=reward,
            observation=observation,
            system=system,
            rewards=rewards,
            run_output=output,
            failed_metrics=["bandwidth"] if case == "metric-failure" else [],
            execution_error="runner failed" if case == "runner-failure" else None,
        )
        loaded = Trajectory.load(trajectory.output_path).records()[0]
        assert loaded.get("run_details", decode_numbers=True)["resolved_action"] == {"x": 1, "y": 10}
        assert loaded["action"] == {"x": 1}
        assert record.get("measurement", decode_numbers=True) == loaded.get("measurement", decode_numbers=True)
        record = loaded
    if case.startswith("malformed-"):
        assert trajectory is not None
        archive = tmp_path / "malformed"
        archive.mkdir()
        (archive / Trajectory.file_name).write_bytes(trajectory.output_path.read_bytes())
        with trajectory.metadata_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        measurement = json.loads(rows[0]["measurement"])
        if case == "malformed-outcome":
            measurement["outcome"] = []
        elif case == "malformed-execution-status":
            measurement["execution_status"] = []
        elif case == "malformed-metric-status":
            measurement["metrics"]["0"]["status"] = []
        else:
            rows[0]["evaluation_failure"] = json.dumps({"metric_indices": "[]"})
        rows[0]["measurement"] = json.dumps(measurement)
        with (archive / Trajectory.metadata_file_name).open("w", newline="") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            csv_writer.writeheader()
            csv_writer.writerows(rows)
        record = Trajectory.load(archive).records()[0]
    if case in ("legacy-empty-failure", "legacy-native-status"):
        paths = record.get("run_details", decode_numbers=True)["recovery"]["opaque_context_paths"]
        assert {tuple(path) for path in paths} == {
            ("evaluation_failure", "constraint"),
            ("evaluation_failure", "metric_indices"),
        }
        assert record.get("evaluation_failure", decode_numbers=True) == {"constraint": "False", "metric_indices": "[]"}
    if case == "legacy-native-status":
        archive = tmp_path / "native-status"
        archive.mkdir()
        with (archive / Trajectory.file_name).open("w", newline="") as stream:
            archive_writer = csv.writer(stream)
            archive_writer.writerow(["step", "action", "reward", "observation"])
            archive_writer.writerow([record.step, record["action"], record["reward"], record["observation"]])
        measurement = record.get("measurement", decode_numbers=True)
        measurement["execution_status"] = []
        with (archive / Trajectory.metadata_file_name).open("w", newline="") as stream:
            archive_writer = csv.writer(stream)
            archive_writer.writerow(
                ["step", "record_context_version", "run_details", "measurement", "evaluation_failure"]
            )
            archive_writer.writerow(
                [
                    record.step,
                    "1",
                    json.dumps(record.get("run_details", decode_numbers=True)),
                    json.dumps(measurement),
                    json.dumps(record.get("evaluation_failure", decode_numbers=True)),
                ]
            )
        record = Trajectory.load(archive).records()[0]
    source_before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    before = dict(record)
    for purpose in ("reference", "measurement", "decision", "learning"):
        reasons = record.ineligibility_reasons(purpose, current=eligibility_context if purpose == "learning" else None)
        assert isinstance(reasons, tuple)
        assert blocked[purpose] in reasons if purpose in blocked else reasons == ()
    assert dict(record) == before
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == source_before
    assert record.step == (0 if case == "step-zero" else 7)
    with pytest.raises(ValueError):
        record.ineligibility_reasons(invalid_purpose)
    with pytest.raises(ValueError):
        record.ineligibility_reasons("learning")


@pytest.mark.parametrize(
    "change,reason",
    [
        ("same", None),
        ("allowed-dr", None),
        ("fixed", "configuration_incompatible"),
        ("zero-weight-dr", "environment_incompatible"),
        ("held-value", "action_incompatible"),
        ("boolean-value", "action_incompatible"),
        ("metric-order", "measurement_schema_incompatible"),
        ("reward-policy", "configuration_incompatible"),
        ("launch", "configuration_incompatible"),
        ("malformed-metric", "measurement_schema_incompatible"),
    ],
)
def test_learning_eligibility_checks_actual_values_full_context_and_supported_regimes(
    tmp_path: Path, eligibility_context, change, reason
):
    from copy import deepcopy

    from cloudai.models.output import Metric, Run

    original, system, rewards = eligibility_context
    resolved = original.apply_params_set({"x": 1, "y": 10, "drop": 0.1})
    resolved.step = 3
    writer = Trajectory(iteration_dir=tmp_path / "history")
    record = writer.record(
        test_run=resolved,
        action_space=original.param_space,
        action={"x": 1},
        reward=-1.0,
        observation=[1.0],
        system=system,
        rewards=rewards,
        run_output=Run(
            path=str(resolved.output_path),
            jobid="job",
            status="completed",
            metrics=[Metric(name="bandwidth", value=1.0, unit="GB/s")],
        ),
    )
    record_path = writer.output_path
    if change == "malformed-metric":
        archive = tmp_path / "archive"
        archive.mkdir()
        record_path = archive / Trajectory.file_name
        record_path.write_bytes(writer.output_path.read_bytes())
        with writer.metadata_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        cell = json.loads(rows[0]["measurement"])
        cell["metrics"] = {"0": "unusable"}
        rows[0]["measurement"] = json.dumps(cell)
        with (archive / Trajectory.metadata_file_name).open("w", newline="") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            csv_writer.writeheader()
            csv_writer.writerows(rows)
        record = Trajectory.load(record_path).records()[0]
    current = deepcopy(original)
    current_system = deepcopy(system)
    current_rewards = deepcopy(rewards)
    if change == "fixed":
        current.test.cmd_args.fixed = "different"
    elif change in ("allowed-dr", "zero-weight-dr"):
        current.test.env_params["drop"].weights = {"allowed-dr": [0.0, 1.0], "zero-weight-dr": [1.0, 0.0]}[change]
    elif change == "held-value":
        current.test.cmd_args.y = [20, 30]
    elif change == "boolean-value":
        current.test.cmd_args.x = [True, 2]
    elif change == "metric-order":
        current.test.agent_metrics = ["other", "bandwidth"]
    elif change == "reward-policy":
        current_rewards.metric_failure = -17.0
    elif change == "launch":
        current.num_nodes = 2
    current.step = 99
    for candidate in (record, Trajectory.load(record_path).records()[0]):
        reasons = candidate.ineligibility_reasons("learning", current=(current, current_system, current_rewards))
        assert reason in reasons if reason else reasons == ()
        assert candidate.ineligibility_reasons("reference") == ()
        assert candidate["action"] == {"x": 1}
    assert current.step == 99
