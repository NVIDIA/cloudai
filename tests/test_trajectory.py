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

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import pytest

from cloudai.configurator.trajectory import (
    CsvTrajectoryWriter,
    EnvParamsSample,
    Trajectory,
    TrajectoryEntry,
    TrialResult,
)


@dataclasses.dataclass(frozen=True)
class LoggingMetrics:
    logging_metrics: Mapping[str, float]


@dataclasses.dataclass(frozen=True)
class CacheContext:
    contributes_to_identity: ClassVar[bool] = True
    cache_context: Mapping[str, Any]


def _entry(step: int, action: Mapping[str, Any] | None = None) -> TrajectoryEntry:
    return TrajectoryEntry(
        step=step,
        components=(TrialResult(action=action or {"x": step}, reward=float(step), observation=[step]),),
    )


def _extended_entry(step: int, speed: int, power: float = 600.0) -> TrajectoryEntry:
    return TrajectoryEntry(
        step=step,
        components=(
            TrialResult(action={"x": 1}, reward=float(step), observation=[step]),
            EnvParamsSample({"speed": speed}),
            LoggingMetrics({"gpu_power_watts": power}),
        ),
    )


def test_trajectory_is_an_ordered_sequence() -> None:
    trajectory = Trajectory([_entry(1), _entry(3)])

    assert len(trajectory) == 2
    assert [entry.step for entry in trajectory] == [1, 3]
    assert trajectory[0].step == 1
    assert [entry.step for entry in trajectory[:]] == [1, 3]


def test_trajectory_rejects_non_increasing_steps() -> None:
    trajectory = Trajectory([_entry(2)])

    with pytest.raises(ValueError, match="steps must increase"):
        trajectory.append(step=2, action={"x": 2}, reward=2.0, observation=[2])


def test_writer_failure_does_not_append_entry_to_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_write(_writer: CsvTrajectoryWriter, _record: Mapping[str, object]) -> None:
        raise OSError("write failed")

    monkeypatch.setattr(CsvTrajectoryWriter, "append", fail_write)
    trajectory = Trajectory(iteration_dir=tmp_path)

    with pytest.raises(OSError, match="write failed"):
        trajectory.append(step=1, action={"x": 1}, reward=1.0, observation=[1])

    assert len(trajectory) == 0


def test_initial_entries_are_not_replayed_to_writer(tmp_path: Path) -> None:
    trajectory = Trajectory([_entry(1)], iteration_dir=tmp_path)

    assert len(trajectory) == 1
    assert not (tmp_path / "trajectory.csv").exists()


def test_append_writes_component_values_to_csv(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    trajectory = Trajectory(
        iteration_dir=tmp_path,
        file_type="csv",
        components=(LoggingMetrics,),
    )

    trajectory.append(step=1, action={"x": 1}, reward=1.0, observation=[1], logging_metrics={"power": 600.0})
    trajectory.append(step=2, action={"x": 2}, reward=2.0, observation=[2], logging_metrics={"power": 610.0})

    assert path.read_text().splitlines() == [
        "step,action,reward,observation,logging_metrics",
        "1,{'x': 1},1.0,[1],{'power': 600.0}",
        "2,{'x': 2},2.0,[2],{'power': 610.0}",
    ]


def test_csv_writer_initializes_a_precreated_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    path.touch()

    Trajectory(iteration_dir=tmp_path).append(step=1, action={"x": 1}, reward=1.0, observation=[1])

    assert path.read_text().splitlines() == [
        "step,action,reward,observation",
        "1,{'x': 1},1.0,[1]",
    ]


def test_csv_writer_reuses_a_matching_header_without_duplication(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    path.write_text("step,action,reward,observation\n")

    Trajectory(iteration_dir=tmp_path).append(step=1, action={"x": 1}, reward=1.0, observation=[1])

    assert path.read_text().splitlines() == [
        "step,action,reward,observation",
        "1,{'x': 1},1.0,[1]",
    ]


def test_csv_writer_rejects_an_existing_mismatched_header(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    path.write_text("step,reward,action,observation\n")

    with pytest.raises(ValueError, match="trajectory file fields do not match"):
        Trajectory(iteration_dir=tmp_path).append(step=1, action={"x": 1}, reward=1.0, observation=[1])

    assert path.read_text() == "step,reward,action,observation\n"


def test_append_writes_generic_records_as_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    trajectory = Trajectory(
        iteration_dir=tmp_path,
        file_type="jsonl",
        components=(EnvParamsSample, LoggingMetrics),
    )

    trajectory.append(
        step=1,
        action={"x": 1},
        reward=1.0,
        observation=[1],
        env_params={"speed": 8},
        logging_metrics={"gpu_power_watts": 610.0},
    )

    assert json.loads(path.read_text()) == {
        "step": 1,
        "action": {"x": 1},
        "reward": 1.0,
        "observation": [1],
        "env_params": {"speed": 8},
        "logging_metrics": {"gpu_power_watts": 610.0},
    }


def test_entry_contains_and_retrieves_typed_components() -> None:
    result = TrialResult({"x": 1}, 1.0, [1])
    env_params = EnvParamsSample({"speed": 1})
    metrics = LoggingMetrics({"gpu_power_watts": 600.0})
    entry = TrajectoryEntry(step=1, components=(result, env_params, metrics))

    assert entry.components == (result, env_params, metrics)
    assert entry.get(TrialResult) is result
    assert entry.get(EnvParamsSample) is not env_params
    assert entry.get(LoggingMetrics) is metrics


def test_entry_rejects_duplicate_component_types() -> None:
    with pytest.raises(ValueError, match="duplicate component types"):
        TrajectoryEntry(
            step=1,
            components=(EnvParamsSample({"speed": 1}), EnvParamsSample({"speed": 2})),
        )


def test_trajectory_validates_its_fixed_component_schema() -> None:
    trajectory = Trajectory(components=(EnvParamsSample, LoggingMetrics))

    with pytest.raises(TypeError, match="missing: logging_metrics"):
        trajectory.append(step=1, action={"x": 1}, reward=1.0, observation=[1], env_params={"speed": 1})

    with pytest.raises(TypeError, match="unexpected: logging_metrics"):
        Trajectory().append(
            step=1,
            action={"x": 1},
            reward=1.0,
            observation=[1],
            logging_metrics={},
        )


def test_find_uses_only_components_that_contribute_to_identity() -> None:
    trajectory = Trajectory(
        components=(EnvParamsSample, LoggingMetrics),
    )
    first = trajectory.append(
        step=1,
        action={"x": 1},
        reward=1.0,
        observation=[1],
        env_params={"speed": 1},
        logging_metrics={"gpu_power_watts": 600.0},
    )

    assert trajectory.find({"x": 1}, env_params={"speed": 1}) is first
    assert trajectory.find({"x": 1}, env_params={"speed": 2}) is None


def test_future_identity_components_are_frozen_by_trajectory(tmp_path: Path) -> None:
    context = {"hardware": {"gpus": [8]}}
    trajectory = Trajectory(iteration_dir=tmp_path, components=(CacheContext,))

    entry = trajectory.append(
        step=1,
        action={"x": 1},
        reward=1.0,
        observation=[1],
        cache_context=context,
    )
    context["hardware"]["gpus"].append(16)
    stored_context = entry.get(CacheContext)
    assert stored_context is not None

    with pytest.raises(TypeError):
        stored_context.cache_context["hardware"] = {}  # type: ignore[index]
    with pytest.raises(AttributeError):
        stored_context.cache_context["hardware"]["gpus"].append(16)

    assert trajectory.find({"x": 1}, cache_context={"hardware": {"gpus": [8]}}) is entry
    assert trajectory.find({"x": 1}, cache_context=context) is None
    assert "16" not in (tmp_path / "trajectory.csv").read_text()


def test_initial_entries_snapshot_future_identity_components() -> None:
    context = CacheContext({"hardware": {"gpus": [8]}})
    entry = TrajectoryEntry(
        step=1,
        components=(TrialResult(action={"x": 1}, reward=1.0, observation=[1]), context),
    )
    trajectory = Trajectory([entry], components=(CacheContext,))

    context.cache_context["hardware"]["gpus"].append(16)

    assert trajectory.find({"x": 1}, cache_context={"hardware": {"gpus": [8]}}) is trajectory[0]


def test_find_ignores_informational_component_values() -> None:
    trajectory = Trajectory(components=(LoggingMetrics,))
    first = trajectory.append(
        step=1,
        action={"x": 1},
        reward=1.0,
        observation=[1],
        logging_metrics={"gpu_power_watts": 600.0},
    )

    assert trajectory.find({"x": 1}) is first


def test_find_requires_the_configured_identity_component_types() -> None:
    trajectory = Trajectory(
        components=(EnvParamsSample,),
    )

    with pytest.raises(TypeError, match="missing: env_params"):
        trajectory.find({"x": 1})

    with pytest.raises(TypeError, match="unexpected: logging_metrics"):
        trajectory.find({"x": 1}, logging_metrics={})


def test_find_preserves_exact_value_types_inside_components() -> None:
    trajectory = Trajectory(
        components=(EnvParamsSample,),
    )
    trajectory.append(step=1, action={"x": 1}, reward=1.0, observation=[1], env_params={"speed": 1.0})

    assert trajectory.find({"x": 1}, env_params={"speed": 1}) is None


def test_find_preserves_exact_action_value_types() -> None:
    trajectory = Trajectory([_entry(1, {"x": 1.0})])

    assert trajectory.find({"x": 1}) is None


def test_identity_values_are_deeply_immutable_and_owned_by_the_entry(tmp_path: Path) -> None:
    trajectory = Trajectory(iteration_dir=tmp_path, components=(EnvParamsSample,))
    action = {"shape": {"layers": [1, 2]}}
    sampled_env_params = {"regime": {"speeds": [3, 4]}}
    entry = trajectory.append(
        step=1,
        action=action,
        reward=1.0,
        observation=[1],
        env_params=sampled_env_params,
    )
    result = entry.get(TrialResult)
    env_params = entry.get(EnvParamsSample)
    assert result is not None and env_params is not None

    action["shape"]["layers"].append(99)
    sampled_env_params["regime"]["speeds"].append(99)
    with pytest.raises(TypeError):
        result.action["shape"] = {}  # type: ignore[index]
    with pytest.raises(AttributeError):
        result.action["shape"]["layers"].append(99)
    with pytest.raises(TypeError):
        env_params.env_params["regime"] = {}  # type: ignore[index]
    with pytest.raises(AttributeError):
        env_params.env_params["regime"]["speeds"].append(99)

    original_action = {"shape": {"layers": [1, 2]}}
    original_env_params = {"regime": {"speeds": [3, 4]}}
    assert trajectory.find(original_action, env_params=original_env_params) is entry
    assert "99" not in (tmp_path / "trajectory.csv").read_text()


def test_trajectory_logs_lifecycle_and_lookup(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("DEBUG"):
        trajectory = Trajectory()
        entry = trajectory.append(step=1, action={"x": 1}, reward=1.0, observation=[1])
        assert trajectory.find({"x": 1}) is entry
        assert trajectory.find({"x": 2}) is None

    assert (
        "Initialized Trajectory with 0 warm-start entries and persistence to local trajectory.csv. "
        "Entries contain component types: [TrialResult]."
    ) in caplog.messages
    assert "Appended trajectory entry for step 1 (total entries: 1)." in caplog.messages
    assert "Found matching trajectory entry at step 1 for action {'x': 1}." in caplog.messages
    assert "No matching trajectory entry found for action {'x': 2}." in caplog.messages
