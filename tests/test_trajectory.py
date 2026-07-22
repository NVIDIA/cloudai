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

from ast import literal_eval
from pathlib import Path

import pandas as pd
import pytest

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


def test_trajectory_rejects_invalid_or_non_increasing_steps(tmp_path: Path) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path)

    with pytest.raises(ValueError, match="positive integer"):
        trajectory.append(step=0, action={"x": 1}, reward=1.0, observation={"metric": 1.0})

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
