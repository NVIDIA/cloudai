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

"""Ordered trajectory steps composed from typed dataclass components."""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Protocol, TypeVar, cast, overload

ComponentT = TypeVar("ComponentT")


@dataclasses.dataclass(frozen=True)
class EnvParamsSample:
    """Environment-parameter values sampled for one trial."""

    contributes_to_identity: ClassVar[bool] = True

    env_params: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Own an immutable snapshot of the sampled values."""
        object.__setattr__(self, "env_params", _freeze(self.env_params))


@dataclasses.dataclass(frozen=True)
class TrialResult:
    """The action and resulting values required for every trajectory step."""

    action: Mapping[str, Any]
    reward: float
    observation: Sequence[Any]

    def __post_init__(self) -> None:
        """Own an immutable snapshot of the action."""
        object.__setattr__(self, "action", _freeze(self.action))


@dataclasses.dataclass(frozen=True)
class TrajectoryEntry:
    """One immutable step containing a fixed set of typed data components."""

    step: int
    components: tuple[object, ...]

    def __post_init__(self) -> None:
        """Validate the trial index and component uniqueness."""
        if self.step < 1:
            raise ValueError(f"trajectory step must be positive; got {self.step}")
        _components_by_type(self.components)

    def get(self, component_type: type[ComponentT]) -> ComponentT | None:
        """Return the component with exactly the requested type, if present."""
        for component in self.components:
            if type(component) is component_type:
                return cast(ComponentT, component)
        return None


class TrajectoryWriter(Protocol):
    """Persistence boundary for one flattened trajectory record."""

    @property
    def output_path(self) -> Path: ...

    def append(self, record: Mapping[str, object]) -> None:
        """Persist one record."""


class _FileTrajectoryWriter:
    """Resolve a writer-specific filename beneath an iteration directory."""

    file_name = ""

    def __init__(self, iteration_dir: Path | Callable[[], Path]) -> None:
        self._iteration_dir = iteration_dir

    @property
    def output_path(self) -> Path:
        iteration_dir = self._iteration_dir() if callable(self._iteration_dir) else self._iteration_dir
        return iteration_dir / self.file_name


class CsvTrajectoryWriter(_FileTrajectoryWriter):
    """Append trajectory records to trajectory.csv."""

    file_name = "trajectory.csv"

    def __init__(self, iteration_dir: Path | Callable[[], Path]) -> None:
        super().__init__(iteration_dir)
        self._fields: tuple[str, ...] | None = None

    def append(self, record: Mapping[str, object]) -> None:
        fields = tuple(record)
        path = self.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        existing_fields: tuple[str, ...] = ()
        if not write_header:
            with path.open(newline="") as file:
                existing_fields = tuple(next(csv.reader(file), ()))

        if self._fields is None:
            if existing_fields and existing_fields != fields:
                raise ValueError(f"trajectory file fields do not match: expected {fields}, got {existing_fields}")
            self._fields = fields
        elif fields != self._fields:
            raise ValueError(f"trajectory record fields changed: expected {self._fields}, got {fields}")
        elif existing_fields and existing_fields != self._fields:
            raise ValueError(f"trajectory file fields do not match: expected {self._fields}, got {existing_fields}")

        with path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self._fields)
            if write_header:
                writer.writeheader()
            writer.writerow(record)
        logging.debug("Wrote trajectory record to %s.", path)


class JsonLinesTrajectoryWriter(_FileTrajectoryWriter):
    """Append trajectory records to trajectory.jsonl as newline-delimited JSON objects."""

    file_name = "trajectory.jsonl"

    def append(self, record: Mapping[str, object]) -> None:
        path = self.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as file:
            file.write(json.dumps(record))
            file.write("\n")
        logging.debug("Wrote trajectory record to %s.", path)


class Trajectory(Sequence[TrajectoryEntry]):
    """
    Ordered entries for one DSE iteration with one fixed component schema.

    ``components`` declares optional data types every entry must contain;
    :class:`TrialResult` is always included.
    Components whose ``contributes_to_identity`` class property is true affect
    trial equivalence; other components are stored as informational data.

    Steps must be appended in increasing order, but gaps are permitted because
    CloudAI does not record constraint-failed trials.
    """

    def __init__(
        self,
        entries: Sequence[TrajectoryEntry] = (),
        *,
        iteration_dir: Path | Callable[[], Path] | None = None,
        file_type: Literal["csv", "jsonl"] = "csv",
        components: Sequence[type[object]] = (),
    ) -> None:
        self._component_types = (TrialResult, *components)
        self._components = frozenset(self._component_types)
        self._identity = frozenset(
            component_type
            for component_type in self._component_types
            if getattr(component_type, "contributes_to_identity", False)
        )
        if len(self._components) != len(self._component_types):
            raise ValueError("components cannot contain duplicate types")

        self._writer = self._create_writer(iteration_dir, file_type)
        self._fields_by_type = self._build_fields_by_type()

        self._entries: list[TrajectoryEntry] = []
        for entry in entries:
            self._store_entry(entry, persist=False)

        component_names = ", ".join(component_type.__name__ for component_type in self._component_types)
        logging.debug(
            "Initialized Trajectory with component types %s and %s entries.",
            component_names,
            len(self),
        )

    @staticmethod
    def _create_writer(
        iteration_dir: Path | Callable[[], Path] | None,
        file_type: Literal["csv", "jsonl"],
    ) -> TrajectoryWriter | None:
        if file_type == "csv":
            writer_type = CsvTrajectoryWriter
        elif file_type == "jsonl":
            writer_type = JsonLinesTrajectoryWriter
        else:
            raise ValueError(f"Invalid trajectory file type: {file_type}")
        return writer_type(iteration_dir) if iteration_dir is not None else None

    @overload
    def __getitem__(self, index: int) -> TrajectoryEntry: ...

    @overload
    def __getitem__(self, index: slice) -> list[TrajectoryEntry]: ...

    def __getitem__(self, index: int | slice) -> TrajectoryEntry | list[TrajectoryEntry]:
        """Return one entry or a list containing an entry slice."""
        return self._entries[index]

    def __len__(self) -> int:
        """Return the number of recorded entries."""
        return len(self._entries)

    def __iter__(self) -> Iterator[TrajectoryEntry]:
        """Iterate over entries in step order."""
        return iter(self._entries)

    @property
    def output_path(self) -> Path | None:
        """Return the writer's current output path, if persistence is configured."""
        return self._writer.output_path if self._writer is not None else None

    def append(self, *, step: int, **values: object) -> TrajectoryEntry:
        """Build configured components from values, then store and persist one entry."""
        components = self._construct_components(self._component_types, values, "trajectory values")
        entry = TrajectoryEntry(step=step, components=components)
        self._store_entry(entry, persist=True)
        return entry

    def _store_entry(self, entry: TrajectoryEntry, *, persist: bool) -> None:
        self._validate_entry_components(entry)
        if self._entries and entry.step <= self._entries[-1].step:
            raise ValueError(f"trajectory steps must increase: last step is {self._entries[-1].step}, got {entry.step}")
        if persist and self._writer is not None:
            self._writer.append(self._to_record(entry))
        self._entries.append(entry)
        logging.debug("Appended trajectory entry for step %s (total entries: %s).", entry.step, len(self))

    def find(self, action: Mapping[str, Any], **identity_values: object) -> TrajectoryEntry | None:
        identity_components = self._construct_components(
            tuple(component_type for component_type in self._component_types if component_type in self._identity),
            identity_values,
            "trajectory identity values",
        )
        identity = self._identity_for(identity_components)
        frozen_action = _freeze(action)
        for entry in self._entries:
            result = entry.get(TrialResult)
            if result is None:
                raise ValueError(f"trajectory entry at step {entry.step} is missing TrialResult")
            if _values_match_exact(result.action, frozen_action) and _values_match_exact(
                self._identity_for(entry.components), identity
            ):
                logging.debug("Found matching trajectory entry at step %s for action %s.", entry.step, action)
                return entry
        logging.debug("No matching trajectory entry found for action %s.", action)
        return None

    def _validate_entry_components(self, entry: TrajectoryEntry) -> None:
        _validate_schema(
            frozenset(type(component) for component in entry.components),
            self._components,
            "trajectory entry components",
        )

    def _identity_for(self, components: Sequence[object]) -> dict[type[object], object]:
        return {type(component): component for component in components if type(component) in self._identity}

    def _construct_components(
        self,
        component_types: Sequence[type[object]],
        values: Mapping[str, object],
        context: str,
    ) -> tuple[object, ...]:
        fields_by_type = {
            component_type: tuple(field for field in self._fields_by_type[component_type] if field.init)
            for component_type in component_types
        }
        expected = {field.name for fields in fields_by_type.values() for field in fields}
        required = {
            field.name
            for fields in fields_by_type.values()
            for field in fields
            if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
        }
        actual = set(values)
        missing = required - actual
        unexpected = actual - expected
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing: {', '.join(sorted(missing))}")
            if unexpected:
                details.append(f"unexpected: {', '.join(sorted(unexpected))}")
            raise TypeError(f"{context} do not match configured schema ({'; '.join(details)})")

        components = []
        for component_type, fields in fields_by_type.items():
            kwargs = {field.name: values[field.name] for field in fields if field.name in values}
            constructor = cast(Any, component_type)
            components.append(constructor(**kwargs))
        return tuple(components)

    def _build_fields_by_type(self) -> dict[type[object], tuple[dataclasses.Field[Any], ...]]:
        fields_by_type: dict[type[object], tuple[dataclasses.Field[Any], ...]] = {}
        for component_type in self._component_types:
            if not dataclasses.is_dataclass(component_type):
                raise TypeError(f"trajectory component type {component_type.__name__} must be a dataclass")
            fields_by_type[component_type] = dataclasses.fields(component_type)

        fields = ("step", *(field.name for component_fields in fields_by_type.values() for field in component_fields))
        if len(fields) != len(set(fields)):
            raise ValueError("trajectory record fields must be unique")
        return fields_by_type

    def _to_record(self, entry: TrajectoryEntry) -> dict[str, object]:
        record: dict[str, object] = {"step": entry.step}
        components_by_type = {type(component): component for component in entry.components}
        for component_type in self._component_types:
            component = components_by_type[component_type]
            record.update(
                {_field.name: _thaw(getattr(component, _field.name)) for _field in self._fields_by_type[component_type]}
            )
        return record


def _components_by_type(components: Sequence[object]) -> dict[type[object], object]:
    non_dataclasses = [type(component).__name__ for component in components if not dataclasses.is_dataclass(component)]
    if non_dataclasses:
        raise TypeError(f"trajectory components must be dataclass instances: {', '.join(non_dataclasses)}")
    by_type = {type(component): component for component in components}
    if len(by_type) != len(components):
        raise ValueError("components cannot contain duplicate component types")
    return by_type


def _validate_schema(
    actual: frozenset[type[object]],
    expected: frozenset[type[object]],
    context: str,
) -> None:
    if actual == expected:
        return

    details = []
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        details.append(f"missing: {', '.join(sorted(component_type.__name__ for component_type in missing))}")
    if unexpected:
        details.append(f"unexpected: {', '.join(sorted(component_type.__name__ for component_type in unexpected))}")
    raise TypeError(f"{context} do not match configured schema ({'; '.join(details)})")


def _freeze(value: Any) -> Any:
    """Recursively copy mutable containers into read-only equivalents."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Convert frozen containers to values supported by trajectory writers."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return [_thaw(item) for item in value]
    return value


def _values_match_exact(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if dataclasses.is_dataclass(left) and not isinstance(left, type):
        return all(
            _values_match_exact(getattr(left, field.name), getattr(right, field.name))
            for field in dataclasses.fields(left)
        )
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return False
        return all(_values_match_exact(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _values_match_exact(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right
