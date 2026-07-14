# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from cloudai.configurator.trajectory import (
    CsvTrajectoryWriter,
    EnvParamsSample,
    JsonLinesTrajectoryWriter,
    Trajectory,
    TrajectoryEntry,
    TrialResult,
)


@dataclasses.dataclass(frozen=True)
class LoggingMetrics:
    logging_metrics: Mapping[str, float]


class RecordingWriter:
    def __init__(self, output_path: Path, *, fail: bool = False) -> None:
        self.output_path = output_path
        self.fail = fail
        self.records: list[Mapping[str, object]] = []

    def append(self, record: Mapping[str, object]) -> None:
        if self.fail:
            raise OSError("write failed")
        self.records.append(record)


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


def test_writer_failure_does_not_append_entry_to_memory(tmp_path: Path) -> None:
    trajectory = Trajectory(writer=RecordingWriter(tmp_path / "trajectory", fail=True))

    with pytest.raises(OSError, match="write failed"):
        trajectory.append(step=1, action={"x": 1}, reward=1.0, observation=[1])

    assert len(trajectory) == 0


def test_initial_entries_are_not_replayed_to_writer(tmp_path: Path) -> None:
    writer = RecordingWriter(tmp_path / "trajectory")

    trajectory = Trajectory([_entry(1)], writer=writer)

    assert len(trajectory) == 1
    assert writer.records == []


def test_append_writes_component_values_to_csv(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    trajectory = Trajectory(
        writer=CsvTrajectoryWriter(tmp_path),
        components=(LoggingMetrics,),
    )

    trajectory.append(step=1, action={"x": 1}, reward=1.0, observation=[1], logging_metrics={"power": 600.0})
    trajectory.append(step=2, action={"x": 2}, reward=2.0, observation=[2], logging_metrics={"power": 610.0})

    assert path.read_text().splitlines() == [
        "step,action,reward,observation,logging_metrics",
        "1,{'x': 1},1.0,[1],{'power': 600.0}",
        "2,{'x': 2},2.0,[2],{'power': 610.0}",
    ]


def test_append_writes_generic_records_as_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    trajectory = Trajectory(
        writer=JsonLinesTrajectoryWriter(tmp_path),
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
    assert entry.get(EnvParamsSample) is env_params
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


def test_identity_component_types_must_be_declared() -> None:
    with pytest.raises(ValueError, match=r"must be declared.*EnvParamsSample"):
        Trajectory(identity=(EnvParamsSample,))


def test_find_uses_only_configured_identity_components() -> None:
    trajectory = Trajectory(
        components=(EnvParamsSample, LoggingMetrics),
        identity=(EnvParamsSample,),
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
        identity=(EnvParamsSample,),
    )

    with pytest.raises(TypeError, match="missing: env_params"):
        trajectory.find({"x": 1})

    with pytest.raises(TypeError, match="unexpected: logging_metrics"):
        trajectory.find({"x": 1}, logging_metrics={})


def test_find_preserves_exact_value_types_inside_components() -> None:
    trajectory = Trajectory(
        components=(EnvParamsSample,),
        identity=(EnvParamsSample,),
    )
    trajectory.append(step=1, action={"x": 1}, reward=1.0, observation=[1], env_params={"speed": 1.0})

    assert trajectory.find({"x": 1}, env_params={"speed": 1}) is None


def test_find_preserves_exact_action_value_types() -> None:
    trajectory = Trajectory([_entry(1, {"x": 1.0})])

    assert trajectory.find({"x": 1}) is None


def test_trajectory_logs_lifecycle_and_lookup(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("DEBUG"):
        trajectory = Trajectory()
        entry = trajectory.append(step=1, action={"x": 1}, reward=1.0, observation=[1])
        assert trajectory.find({"x": 1}) is entry
        assert trajectory.find({"x": 2}) is None

    assert "Initialized Trajectory with component types TrialResult and 0 entries." in caplog.messages
    assert "Appended trajectory entry for step 1 (total entries: 1)." in caplog.messages
    assert "Found matching trajectory entry at step 1 for action {'x': 1}." in caplog.messages
    assert "No matching trajectory entry found for action {'x': 2}." in caplog.messages
