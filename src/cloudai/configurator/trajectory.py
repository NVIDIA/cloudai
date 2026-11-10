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

"""Pandas-native trajectory storage with flat, namespaced columns."""

from __future__ import annotations

import ast
import csv
import io
import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd


class TrajectoryLoadError(RuntimeError):  # noqa: D101
    def __init__(self, path: Path, reason: str, *, row: int | None = None, step: int | None = None) -> None:
        location = f"row {row}, " if row is not None else ""
        location += f"step {step}, " if step is not None else ""
        super().__init__(f"Cannot load trajectory {path}: {location}{reason}")


class _NonfiniteLiteralNames(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in ("nan", "inf"):
            return ast.copy_location(ast.Constant(float(node.id)), node)
        return node


@dataclass(frozen=True, eq=False)
class TrajectoryRecord(Mapping[str, Any]):  # noqa: D101
    step: int
    source: Path
    _domains: Mapping[str, Any]
    row: int | None = None

    def __post_init__(self) -> None:  # noqa: D105
        object.__setattr__(self, "_domains", deepcopy(self._domains))

    def __getitem__(self, key: str) -> Any:  # noqa: D105
        return self.step if key == "step" else self._domain(key)

    def __iter__(self) -> Iterator[str]:  # noqa: D105
        yield "step"
        yield from self._domains

    def __len__(self) -> int:  # noqa: D105
        return len(self._domains) + 1

    def get(self, key: str, default: Any = None, *, decode_numbers: bool = False) -> Any:
        if key == "step":
            return self.step

        try:
            return self._domain(key, decode_numbers=decode_numbers)
        except KeyError:
            return default

    def _domain(self, name: str, *, decode_numbers: bool = False) -> Any:
        value = deepcopy(self._domains[name])
        return _decode_numbers(value) if decode_numbers else value

    def format_observation(self) -> str:
        try:
            observation = self._domain("observation")
        except KeyError:
            return "n/a"
        if isinstance(observation, str) and observation.startswith("[truncated"):
            return "n/a"

        values = observation if isinstance(observation, list) else [observation]
        parts = [f"{value:.4f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value) for value in values]
        return ", ".join(parts) if parts else "n/a"

    @property
    def metadata(self) -> dict[str, Any]:
        return {key: deepcopy(value) for key, value in self._domains.items() if key not in Trajectory._core_domains}


class Trajectory:
    """An ordered DataFrame of DSE steps persisted as core and metadata CSVs."""

    file_name = "trajectory.csv"
    metadata_file_name = "metadata.csv"
    _core_fields = ("step", "action", "reward", "observation")
    _core_domains = frozenset(_core_fields)

    def __init__(
        self,
        *,
        iteration_dir: Path,
        dataframe: pd.DataFrame | None = None,
    ) -> None:
        self._iteration_dir = iteration_dir
        self._dataframe = lazy.pd.DataFrame() if dataframe is None else _copy_dataframe(dataframe)
        self._validate_dataframe()
        self._dataframe = self._dataframe.astype(object)
        # Rows are authoritative; the DataFrame is rebuilt on demand. Appending by
        # concatenating a one-row frame copies every accumulated row (O(N^2) over a run),
        # and ``find`` scanning with ``iterrows`` is O(N) per call, so a cached lookup index
        # is kept alongside. Both are invisible when a trial costs seconds and dominate
        # once trials are cheap.
        self._rows: list[dict[str, Any]] = (
            [dict(record) for _, record in self._dataframe.iterrows()] if len(self._dataframe) else []
        )
        self._find_index: dict[tuple[str, ...], dict[tuple, list[int]]] = {}
        self._frame_dirty = False

        self._source_csv_dataframe: pd.DataFrame | None = None
        self._loaded_records: tuple[TrajectoryRecord, ...] | None = None
        self._source_path: Path | None = None
        self._source_metadata_path: Path | None = None

        logging.debug(
            "Initializing Trajectory: entries=%s, columns=%s.",
            len(self),
            list(self._dataframe.columns),
        )

    @classmethod
    def load(cls, source: str | Path) -> Trajectory:
        source = Path(source)
        path = source / cls.file_name if source.is_dir() else source
        columns, rows, text = _read_csv(path, required=cls._core_fields)

        sidecar = path.parent / cls.metadata_file_name
        has_metadata = sidecar.exists() or sidecar.is_symlink()
        metadata_rows = cls._read_metadata(sidecar) if has_metadata else None

        records, flattened_rows = cls._load_records(path, rows, sidecar, metadata_rows)
        frame = lazy.pd.DataFrame(flattened_rows, dtype=object)
        if not flattened_rows:
            frame = lazy.pd.DataFrame(columns=columns)

        loaded = cls(iteration_dir=path.parent)
        loaded._dataframe = _copy_dataframe(frame)
        loaded._rows = deepcopy(flattened_rows)
        loaded._source_path = path
        loaded._source_metadata_path = sidecar
        loaded._loaded_records = tuple(records)

        try:
            loaded._source_csv_dataframe = lazy.pd.read_csv(io.StringIO(text))
        except (ValueError, lazy.pd.errors.ParserError) as error:
            raise TrajectoryLoadError(path, f"cannot construct CSV view: {error}") from error

        return loaded

    @classmethod
    def _read_metadata(cls, path: Path) -> dict[int, tuple[int, dict[str, str]]]:
        columns, rows, _ = _read_csv(path, required=("step",))
        for column in columns:
            if column != "step" and column.split(".", maxsplit=1)[0] in cls._core_domains:
                raise TrajectoryLoadError(path, f"metadata overlaps core domain: {column}", row=1)

        entries = {}
        for row_number, row in rows:
            step = _read_step(row["step"], path, row_number)
            if step in entries:
                raise TrajectoryLoadError(path, "duplicate metadata step", row=row_number, step=step)
            entries[step] = row_number, row

        return entries

    @classmethod
    def _load_records(
        cls,
        path: Path,
        rows: list[tuple[int, dict[str, str]]],
        sidecar: Path,
        metadata_rows: dict[int, tuple[int, dict[str, str]]] | None,
    ) -> tuple[list[TrajectoryRecord], list[dict[str, Any]]]:
        records = []
        flattened_rows = []
        previous_step = -1
        steps = set()

        for row_number, row in rows:
            step = _read_step(row["step"], path, row_number)
            if step <= previous_step:
                raise TrajectoryLoadError(
                    path, "steps must be unique and strictly increasing", row=row_number, step=step
                )
            previous_step = step
            steps.add(step)

            record = cls._build_record(path, row_number, row, step=step, sidecar=sidecar, metadata_rows=metadata_rows)
            flattened: dict[str, Any] = {"step": step}
            for name, value in record._domains.items():
                flattened.update(_flatten_value(name, value))
            flattened_rows.append(flattened)
            records.append(record)

        unmatched = set(metadata_rows or {}) - steps
        if unmatched:
            step = min(unmatched)
            raise TrajectoryLoadError(
                sidecar, "metadata step has no trajectory row", row=(metadata_rows or {})[step][0], step=step
            )

        return records, flattened_rows

    @classmethod
    def _build_record(
        cls,
        path: Path,
        row_number: int,
        row: dict[str, str],
        *,
        step: int,
        sidecar: Path,
        metadata_rows: dict[int, tuple[int, dict[str, str]]] | None,
    ) -> TrajectoryRecord:
        domains = {
            name: _decode_core_value(row[name], path, row_number, name)
            for name in cls._core_fields
            if name != "step" and name in row
        }
        extra = {name: value for name, value in row.items() if name not in cls._core_fields}
        source_paths = {name: path for name in extra}

        if metadata_rows is not None:
            if step not in metadata_rows:
                raise TrajectoryLoadError(sidecar, "missing metadata row for trajectory step", step=step)
            metadata_row_number, metadata_row = metadata_rows[step]
            for name, value in metadata_row.items():
                if name != "step":
                    if name in extra:
                        raise TrajectoryLoadError(
                            sidecar, f"duplicate metadata field: {name}", row=metadata_row_number, step=step
                        )
                    extra[name] = value
                    source_paths[name] = sidecar

        for name, value in extra.items():
            _set_domain(domains, name, value, source_paths[name], step)

        return TrajectoryRecord(step, path, domains, row_number)

    def records(self) -> tuple[TrajectoryRecord, ...]:
        if self._loaded_records is not None:
            return deepcopy(self._loaded_records)

        records = []
        for row in self._rows:
            domains: dict[str, Any] = {}
            for name, value in row.items():
                if name != "step":
                    _set_domain(domains, name, value, self.output_path, row["step"])
            records.append(TrajectoryRecord(row["step"], self.output_path, domains))

        return tuple(records)

    @property
    def csv_dataframe(self) -> pd.DataFrame:
        if self._source_csv_dataframe is None:
            raise RuntimeError("CSV source view is only available for a loaded trajectory")
        return self._source_csv_dataframe.copy(deep=True)

    def __len__(self) -> int:
        """Return the number of trajectory rows."""
        return len(self._rows)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return a copy of the trajectory DataFrame for analysis."""
        self._materialize()
        return _copy_dataframe(self._dataframe)

    @property
    def output_path(self) -> Path:
        """Return the trajectory CSV path."""
        return self._source_path if self._source_path is not None else self._iteration_dir / self.file_name

    @property
    def metadata_path(self) -> Path:
        """Return the trajectory metadata CSV path."""
        return (
            self._source_metadata_path
            if self._source_metadata_path is not None
            else self._iteration_dir / self.metadata_file_name
        )

    def append(
        self,
        *,
        step: int,
        action: object,
        reward: object,
        observation: object,
        **values: object,
    ) -> pd.Series:
        """Flatten, persist, and store one trajectory row."""
        _validate_step(step)
        if self._rows and step <= self._rows[-1]["step"]:
            raise ValueError(f"trajectory steps must increase: last step is {self._rows[-1]['step']}, got {step}")

        record: dict[str, object] = {"step": step}
        domains = {"action": action, "reward": reward, "observation": observation, **values}
        for domain, value in domains.items():
            for field, field_value in _flatten_value(domain, value).items():
                if field in record:
                    raise ValueError(f"trajectory values produce duplicate column: {field}")
                record[field] = deepcopy(field_value)

        fields = tuple(record)
        expected_fields = tuple(self._rows[0]) if self._rows else tuple(self._dataframe.columns)
        if expected_fields and fields != expected_fields:
            raise ValueError(f"trajectory record fields changed: expected {expected_fields}, got {fields}")

        row = lazy.pd.Series(record, dtype=object)
        row_trajectory = lazy.pd.Series(
            {
                "step": row["step"],
                "action": action,
                "reward": reward,
                "observation": list(observation.values()) if isinstance(observation, Mapping) else observation,
            },
            dtype=object,
        )
        metadata_fields = tuple(
            field for field in row.index if field == "step" or field.split(".", maxsplit=1)[0] not in self._core_domains
        )
        row_metadata = row[list(metadata_fields)] if len(metadata_fields) > 1 else None

        if row_metadata is not None:
            _validate_csv_header(self.metadata_path, tuple(row_metadata.index))
        _append_csv_row(row_trajectory, self.output_path)
        if row_metadata is not None:
            _append_csv_row(row_metadata, self.metadata_path)

        self._rows.append(record)
        position = len(self._rows) - 1
        for fields, index in self._find_index.items():
            index.setdefault(tuple(_hashable_key(record[field]) for field in fields), []).append(position)
        self._frame_dirty = True
        self._loaded_records = None
        logging.debug("Appended trajectory row for step %s (total rows: %s).", step, len(self))
        return _copy_series(row)

    def find(self, **values: object) -> pd.Series | None:
        """Return a copy of the first row matching all supplied domain values."""
        criteria: dict[str, object] = {}
        for domain, value in values.items():
            for field, field_value in _flatten_value(domain, value).items():
                if field in criteria:
                    raise ValueError(f"trajectory values produce duplicate column: {field}")
                criteria[field] = field_value

        if not self._rows:
            return None
        if any(field not in self._rows[0] for field in criteria):
            return None

        # One index per distinct criteria field-set, built on first use. Callers query with a
        # stable set of domains, so this is built once and maintained by ``append``.
        fields = tuple(sorted(criteria))
        index = self._find_index.get(fields)
        if index is None:
            index = {}
            for position, record in enumerate(self._rows):
                index.setdefault(tuple(_hashable_key(record[field]) for field in fields), []).append(position)
            self._find_index[fields] = index

        # A bucket narrows the scan; it does not prove equality. Distinct values can share a
        # key, so the exact predicate still decides, keeping the pre-index result.
        bucket = index.get(tuple(_hashable_key(criteria[field]) for field in fields), ())
        record = next(
            (
                candidate
                for candidate in (self._rows[position] for position in bucket)
                if all(_values_match_exact(candidate[field], criteria[field]) for field in fields)
            ),
            None,
        )
        if record is None:
            logging.debug("No matching trajectory row found for %s.", values)
            return None
        logging.debug("Found matching trajectory row at step %s for %s.", record["step"], values)
        return _copy_series(lazy.pd.Series(record, dtype=object))

    def _materialize(self) -> None:
        """Rebuild the DataFrame from the row buffer, once, when a caller asks for it."""
        if not self._frame_dirty:
            return
        # dtype=object at construction, not astype afterwards: pandas infers a column dtype
        # first, so a column holding both 1 and 2.5 would widen the int to 1.0 before the
        # cast froze it.
        self._dataframe = lazy.pd.DataFrame(self._rows, dtype=object)
        self._frame_dirty = False

    def _validate_dataframe(self) -> None:
        if not self._dataframe.columns.is_unique:
            raise ValueError("trajectory dataframe columns must be unique")
        non_string_columns = [column for column in self._dataframe.columns if not isinstance(column, str)]
        if non_string_columns:
            raise TypeError(f"trajectory dataframe columns must be strings: {non_string_columns}")
        if self._dataframe.empty and len(self._dataframe.columns) == 0:
            return
        if "step" not in self._dataframe.columns:
            raise ValueError("trajectory dataframe must contain a step column")

        previous_step: int | None = None
        for dataframe_step in self._dataframe["step"]:
            step = dataframe_step.item() if hasattr(dataframe_step, "item") else dataframe_step
            _validate_step(step)
            if previous_step is not None and step <= previous_step:
                raise ValueError(f"trajectory steps must increase: last step is {previous_step}, got {step}")
            previous_step = step


def _decode_numbers(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {name: _decode_numbers(child) for name, child in value.items()}
    if isinstance(value, list):
        return [_decode_numbers(child) for child in value]
    if isinstance(value, str):
        for converter in (int, float):
            try:
                return converter(value)
            except ValueError:
                continue

    return value


def _read_csv(path: Path, *, required: Sequence[str]) -> tuple[list[str], list[tuple[int, dict[str, str]]], str]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            text = handle.read()
    except (OSError, UnicodeError) as error:
        raise TrajectoryLoadError(path, f"cannot read artifact: {error}") from error

    if "\x00" in text:
        raise TrajectoryLoadError(path, "artifact contains a NUL character")

    previous_limit = csv.field_size_limit()
    csv.field_size_limit(max(previous_limit, len(text)))
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader, None)
        if not header:
            raise TrajectoryLoadError(path, "missing CSV header", row=1)
        if any(not name for name in header) or len(header) != len(set(header)):
            raise TrajectoryLoadError(path, "CSV header must contain unique nonempty names", row=1)
        missing = set(required) - set(header)
        if missing:
            raise TrajectoryLoadError(path, f"missing required columns: {sorted(missing)}", row=1)

        rows = []
        for row_number, values in enumerate(reader, start=2):
            if len(values) != len(header):
                raise TrajectoryLoadError(path, f"expected {len(header)} fields, got {len(values)}", row=row_number)
            rows.append((row_number, dict(zip(header, values, strict=True))))
    except csv.Error as error:
        raise TrajectoryLoadError(path, f"malformed CSV: {error}", row=reader.line_num) from error
    finally:
        csv.field_size_limit(previous_limit)

    return header, rows, text


def _read_step(value: str, path: Path, row: int) -> int:
    try:
        step = int(value)
    except ValueError as error:
        raise TrajectoryLoadError(path, f"step must be a nonnegative integer, got {value!r}", row=row) from error
    if step < 0:
        raise TrajectoryLoadError(path, f"step must be a nonnegative integer, got {value!r}", row=row, step=step)
    return step


def _decode_core_value(
    value: str,
    path: Path,
    row: int,
    name: str,
) -> Any:
    if name == "observation" and value.startswith("[truncated"):
        return value

    try:
        expression = _NonfiniteLiteralNames().visit(ast.parse(value, mode="eval"))
        return ast.literal_eval(expression)
    except (ValueError, SyntaxError, TypeError, RecursionError) as error:
        if not value or value[0] not in "[({\"'":
            return value
        raise TrajectoryLoadError(path, f"invalid {name} literal: {error}", row=row) from error


def _set_domain(domains: dict[str, Any], name: str, value: Any, path: Path, step: int) -> None:
    parts = name.split(".")
    if any(not part for part in parts):
        raise TrajectoryLoadError(path, f"empty metadata path component: {name}", step=step)

    current = domains
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        child = current[part]
        if not isinstance(child, dict):
            raise TrajectoryLoadError(path, f"conflicting domain path: {name}", step=step)
        current = child

    if parts[-1] in current:
        raise TrajectoryLoadError(path, f"duplicate domain path: {name}", step=step)
    current[parts[-1]] = deepcopy(value)


def _validate_step(step: object) -> None:
    if type(step) is not int or step < 1:
        raise ValueError(f"trajectory step must be a positive integer; got {step}")


def _append_csv_row(row: pd.Series, path: Path) -> None:
    """Validate and append one Series to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _validate_csv_header(path, tuple(row.index))
    row.to_frame().T.to_csv(path, mode="a", header=header, index=False)
    logging.debug("Wrote trajectory row to %s.", path)


def _validate_csv_header(path: Path, fields: tuple[str, ...]) -> bool:
    """Validate an existing CSV header and return whether a header must be written."""
    write_header = not path.exists() or path.stat().st_size == 0
    if write_header:
        return True
    with path.open(newline="") as file:
        existing_fields = tuple(next(csv.reader(file), ()))
    if existing_fields != fields:
        raise ValueError(f"trajectory file fields do not match: expected {fields}, got {existing_fields}")
    return False


def _flatten_value(key: str, value: object) -> dict[str, object]:
    """Flatten a value into dot-separated columns."""
    if isinstance(value, Mapping):
        record: dict[str, object] = {}
        for child_key, child_value in value.items():
            if not isinstance(child_key, str):
                raise TypeError(f"trajectory mapping keys must be strings: {child_key}")
            for field, field_value in _flatten_value(f"{key}.{child_key}", child_value).items():
                if field in record:
                    raise ValueError(f"trajectory values produce duplicate column: {field}")
                record[field] = field_value
        return record
    return {key: value}


def _copy_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Copy a DataFrame and recursively snapshot object cells."""
    copied = dataframe.copy(deep=True).astype(object)
    for row_index in range(dataframe.shape[0]):
        for column_index in range(dataframe.shape[1]):
            copied.iat[row_index, column_index] = deepcopy(dataframe.iat[row_index, column_index])
    return copied


def _copy_series(series: pd.Series) -> pd.Series:
    """Copy a Series and recursively snapshot object cells."""
    copied = series.copy(deep=True).astype(object)
    for index in range(len(series)):
        copied.iat[index] = deepcopy(series.iat[index])
    return copied


def _hashable_key(value: Any) -> Any:
    """
    Return a hashable proxy that groups values :func:`_values_match_exact` would accept.

    The one invariant: values the predicate accepts must never get different keys. A key
    coarser than equality only widens a bucket, which the exact check then filters, but a
    key finer than equality hides a matching row and nothing downstream can recover it.

    So a shared key is not proof of a match, and callers must confirm a candidate with the
    predicate rather than trust the key. The type name is part of the key, keeping ``1``,
    ``1.0`` and ``True`` distinct, which the predicate also requires.
    """
    type_name = type(value).__name__
    if isinstance(value, Mapping):
        # Keys stay themselves, in a frozenset: the predicate compares mappings by key set
        # and order-insensitively, and a frozenset matches both. Stringifying keys would put
        # {1: "a"} and {True: "a"} in different buckets, though the predicate accepts them.
        return (type_name, frozenset((k, _hashable_key(v)) for k, v in value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return (type_name, tuple(_hashable_key(item) for item in value))
    try:
        hash(value)
    except TypeError:
        # One bucket per type, not repr: repr splits values the predicate accepts, such as
        # the sets {1} and {1.0}, and a split is unrecoverable where a shared bucket is not.
        return (type_name,)
    return (type_name, value)


def _values_match_exact(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return False
        return all(_values_match_exact(left[key], right[key]) for key in left)
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        return len(left) == len(right) and all(
            _values_match_exact(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)
