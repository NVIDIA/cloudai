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
import math
from ast import literal_eval
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
import pytest

from cloudai.configurator import Trajectory, TrajectoryLoadError, TrajectoryRecord


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


def test_csv_reuses_a_matching_header_without_duplication(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    path.write_text("step,action,reward,observation\n")

    Trajectory(iteration_dir=tmp_path).append(step=1, action={"x": 1}, reward=1.0, observation={"metric": 2.0})

    assert path.read_text().splitlines() == [
        "step,action,reward,observation",
        "1,{'x': 1},1.0,[2.0]",
    ]


def test_csv_rejects_an_existing_mismatched_header(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    path.write_text("step,reward,action,observation\n")

    with pytest.raises(ValueError, match="file fields do not match"):
        Trajectory(iteration_dir=tmp_path).append(step=1, action={"x": 1}, reward=1.0, observation={"metric": 2.0})

    assert path.read_text() == "step,reward,action,observation\n"


def test_core_only_trajectory_does_not_create_metadata_file(tmp_path: Path) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)

    trajectory.append(step=1, action={"x": 1}, reward=1.0, observation={"metric": 2.0})

    assert not trajectory.metadata_path.exists()


def test_metadata_reuses_matching_header_without_duplication(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text("step,env_params.speed,logging.power\n")
    trajectory = Trajectory(iteration_dir=tmp_path)

    trajectory.append(
        step=1,
        action={"x": 1},
        reward=1.0,
        observation={"metric": 2.0},
        env_params={"speed": 100},
        logging={"power": 600.0},
    )

    assert path.read_text().splitlines() == [
        "step,env_params.speed,logging.power",
        "1,100,600.0",
    ]


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
    metadata_path.write_text("step,logging.power,env_params.speed\n")
    trajectory = Trajectory(iteration_dir=tmp_path)

    with pytest.raises(ValueError, match="file fields do not match"):
        trajectory.append(
            step=1,
            action={"x": 1},
            reward=1.0,
            observation={"metric": 2.0},
            env_params={"speed": 100},
            logging={"power": 600.0},
        )

    assert not trajectory.output_path.exists()
    assert len(trajectory) == 0


def test_persistence_failure_does_not_append_to_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("write failed")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_write)

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
            env_params={"workload": "bisection", "msg_size": 1024, "drop_rate": 0.003},
            execution={"ticket": f"T-{step}"},
        )
    original_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    loaded = Trajectory.load(tmp_path)
    records = loaded.records()

    assert [record.step for record in records] == [1, 2]
    assert [record.domain("action") for record in records] == [{"ooo": 10}, {"ooo": 11}]
    assert [record.domain("reward") for record in records] == [0.1, 0.2]
    assert [record.domain("observation") for record in records] == [[10.0], [20.0]]
    assert records[0].domain("env_params") == {
        "workload": "bisection",
        "msg_size": "1024",
        "drop_rate": "0.003",
    }
    assert records[0].metadata["execution"] == {"ticket": "T-1"}
    assert records[0].source == written.output_path
    assert "observation.bus_bw" not in loaded.dataframe.columns

    found = loaded.find(action={"ooo": 11})
    assert found is not None
    assert found["step"] == 2

    pd.testing.assert_frame_equal(loaded.csv_dataframe, pd.read_csv(written.output_path))
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original_files


def test_load_preserves_source_metadata_until_an_explicit_projection(tmp_path: Path) -> None:
    written = Trajectory(iteration_dir=tmp_path)
    written.append(
        step=1,
        action={"ooo": 10},
        reward=1.0,
        observation=[100.0],
        env_params={"label": "001", "msg_size": 1024, "drop_rate": 0.003},
    )

    loaded = Trajectory.load(tmp_path)
    projected = loaded.records()[0].domain("env_params", decode_numbers=True)

    assert projected == {"label": 1, "msg_size": 1024, "drop_rate": 0.003}
    assert loaded.records()[0].domain("env_params") == {"label": "001", "msg_size": "1024", "drop_rate": "0.003"}


def test_loaded_views_and_records_do_not_share_mutable_state(tmp_path: Path) -> None:
    written = Trajectory(iteration_dir=tmp_path)
    written.append(step=1, action={"values": [1, 2]}, reward=1.0, observation=[2.0], execution={"ticket": "T-1"})

    loaded = Trajectory.load(tmp_path)
    record = loaded.records()[0]

    record.domain("action")["values"].append(3)
    record.metadata["execution"]["ticket"] = "changed"

    flat = loaded.dataframe
    values: object = flat.at[0, "action.values"]
    assert isinstance(values, list)
    values.append(3)

    serialized = loaded.csv_dataframe
    serialized.loc[0, "reward"] = 9.0

    assert loaded.records()[0].domain("action") == {"values": [1, 2]}
    assert loaded.records()[0].metadata["execution"] == {"ticket": "T-1"}

    assert loaded.dataframe.loc[0, "action.values"] == [1, 2]
    assert loaded.csv_dataframe.loc[0, "reward"] == 1.0


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

    assert record.domain("action") == action
    assert live_record.domain("action") == record.domain("action")
    assert record.domain("reward") == reward
    assert live_record.domain("reward") == record.domain("reward")
    assert record.domain("observation") == observation
    assert live_record.domain("observation") == record.domain("observation")
    assert record.metadata == {}
    assert not written.metadata_path.exists()


def test_load_accepts_an_explicit_csv_with_its_sibling_metadata(tmp_path: Path) -> None:
    written = Trajectory(iteration_dir=tmp_path)
    written.append(step=1, action={"ooo": 10}, reward=1.0, observation=[100.0], env_params={"workload": "bisection"})
    csv_path = written.output_path.rename(tmp_path / "measurements.csv")

    record = Trajectory.load(csv_path).records()[0]

    assert record.source == csv_path
    assert record.domain("env_params") == {"workload": "bisection"}


@pytest.mark.parametrize("with_metadata", [False, True])
def test_load_accepts_a_header_only_history(tmp_path: Path, with_metadata: bool) -> None:
    (tmp_path / "trajectory.csv").write_text("step,action,reward,observation\n")
    if with_metadata:
        (tmp_path / "metadata.csv").write_text("step,env_params.workload\n")

    loaded = Trajectory.load(tmp_path)

    assert len(loaded) == 0
    assert loaded.records() == ()
    assert loaded.csv_dataframe.empty


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
    written.metadata_path.write_text("step,env_params.workload\n1,bisection\n")

    with pytest.raises(TrajectoryLoadError) as failure:
        Trajectory.load(tmp_path)

    assert str(written.metadata_path) in str(failure.value)
    assert "step 2" in str(failure.value)
    assert "missing metadata row for trajectory step" in str(failure.value)


def test_load_accepts_an_extensionless_csv_path(tmp_path: Path) -> None:
    written = Trajectory(iteration_dir=tmp_path)
    written.append(step=1, action={"ooo": 10}, reward=1.0, observation=[100.0])
    source = written.output_path.rename(tmp_path / "history")

    record = Trajectory.load(source).records()[0]

    assert record.source == source
    assert record.domain("action") == {"ooo": 10}


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

        assert loaded.records()[0].domain("action") == {"payload": payload}
        assert loaded.records()[0].domain("observation") == [payload]
        assert loaded.records()[0].metadata["logging"] == {"payload": payload}
        pd.testing.assert_frame_equal(loaded.csv_dataframe, pd.read_csv(written.output_path))
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

    for decoded in (record.domain("action")["values"], record.domain("reward")["values"], record.domain("observation")):
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

    assert loaded.records()[0].domain("env_params") == {"note": note}
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original_files


def test_load_preserves_zero_based_legacy_history_and_metadata_identity(tmp_path: Path) -> None:
    core_path = tmp_path / "trajectory.csv"
    metadata_path = tmp_path / "metadata.csv"

    with core_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "action", "reward", "observation", "step.extra"])
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

    assert [(record.step, record.domain("env_params")) for record in records] == [
        (0, {"workload": "thin"}),
        (1, {"workload": "wide"}),
    ]
    assert records[0].domain("action") == {"ooo": 10}
    assert loaded.dataframe["step.extra"].tolist() == ["legacy-0", "legacy-1"]

    found = loaded.find(action={"ooo": 10}, env_params={"workload": "thin"})
    assert found is not None
    assert found["step"] == 0

    pd.testing.assert_frame_equal(loaded.csv_dataframe, pd.read_csv(core_path))
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
    trajectory = Trajectory(iteration_dir=tmp_path)
    trajectory.append(step=1, action={"ooo": 10}, reward=1.0, observation=observation if present else {})
    record = trajectory.records()[0]
    original_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    before = repr(record.domain("observation")) if present else None

    assert record.format_observation() == expected

    if present:
        assert repr(record.domain("observation")) == before
    else:
        with pytest.raises(KeyError):
            record.domain("observation")

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original_files


@pytest.fixture(params=[False, True], ids=["live", "loaded"])
def readonly_record(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[TrajectoryRecord, MappingProxyType[str, Any]]:
    expected = {
        "step": 1,
        "action": {"values": [1, 2]},
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

    assert record.domain("env_params", decode_numbers=True) == {"message_size": 1024}
    assert record["env_params"] == {"message_size": "1024"}


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
        dict(record.items())["action"],
        list(record.values())[1],
        dict(record)["action"],
        {**record}["action"],
    ):
        assert isinstance(detached, dict)
        detached["values"].append(3)
        assert record == expected
