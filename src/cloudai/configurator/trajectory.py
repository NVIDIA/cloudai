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
import hashlib
import io
import json
import logging
import os
import stat
import tempfile
import threading
import uuid
import weakref
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import pandas as pd


class TrajectoryLoadError(RuntimeError):  # noqa: D101
    def __init__(
        self,
        path: Path,
        reason: str,
        *,
        row: int | None = None,
        step: int | None = None,
    ) -> None:
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
        return self.step if key == "step" else deepcopy(self._domains[key])

    def __iter__(self) -> Iterator[str]:  # noqa: D105
        yield "step"
        yield from (name for name in self._domains if name != "step")

    def __len__(self) -> int:  # noqa: D105
        return len(self._domains) + ("step" not in self._domains)

    def get(self, key: str, default: Any = None, *, decode_numbers: bool = False) -> Any:
        try:
            value = self[key]
        except KeyError:
            return default
        return _decode_numbers(value) if decode_numbers else value


class Trajectory:
    """An ordered DataFrame of DSE steps persisted as core and metadata CSVs."""

    file_name = "trajectory.csv"
    metadata_file_name = "metadata.csv"
    _core_fields = ("step", "action", "reward", "observation")
    _core_domains = frozenset(_core_fields)

    _csv_read_lock = threading.Lock()

    def __init__(
        self,
        *,
        iteration_dir: Path,
        dataframe: pd.DataFrame | None = None,
        mode: Literal["create", "resume", "adopt"] = "create",
    ) -> None:
        import fcntl

        if mode not in ("create", "resume", "adopt"):
            raise ValueError(f"unsupported trajectory mode: {mode!r}")
        if dataframe is not None and mode != "create":
            raise ValueError("a warm-start dataframe is supported only when creating a history")
        self._initialize_state(iteration_dir, dataframe)
        _ensure_durable_directory(self._iteration_dir)
        descriptor = os.open(self._iteration_dir / ".trajectory.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(descriptor)
            raise RuntimeError(f"trajectory history already has a writer: {self._iteration_dir}") from error
        self._writer_finalizer = weakref.finalize(self, os.close, descriptor)
        try:
            self._open_writer(mode)
        except BaseException:
            self._writer_finalizer()
            self._writer_finalizer = None
            raise

    @classmethod
    def load(cls, source: str | Path, *, missing_ok: bool = False) -> Trajectory:
        source = Path(source)
        path = source / cls.file_name if source.is_dir() else source
        marker = _read_commit(path.parent)
        try:
            loaded = cls._load_snapshot(path, marker, missing_ok)
        except TrajectoryLoadError:
            if marker is not None:
                raise
            marker = _read_commit(path.parent)
            if marker is None:
                raise
        else:
            if marker is not None:
                return loaded
            marker = _read_commit(path.parent)
            if marker is None:
                return loaded
        return cls._load_snapshot(path, marker, missing_ok)

    @classmethod
    def _read_metadata(
        cls, path: Path, *, text: str | None = None
    ) -> tuple[list[str], dict[int, tuple[int, dict[str, str]]]]:
        columns, rows = _read_csv(path, required=("step",), text=text)
        for column in columns:
            if column != "step" and column.split(".", maxsplit=1)[0] in cls._core_domains:
                raise TrajectoryLoadError(path, f"metadata overlaps core domain: {column}", row=1)

        entries = {}
        for row_number, row in rows:
            step = _read_step(row["step"], path, row_number)
            if step in entries:
                raise TrajectoryLoadError(path, "duplicate metadata step", row=row_number, step=step)
            entries[step] = row_number, row

        return columns, entries

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
                    path,
                    "steps must be unique and strictly increasing",
                    row=row_number,
                    step=step,
                )
            previous_step = step
            steps.add(step)

            record = cls._build_record(
                path,
                row_number,
                row,
                step=step,
                sidecar=sidecar,
                metadata_rows=metadata_rows,
            )
            flattened_rows.append(_flatten_record(record))
            records.append(record)

        unmatched = set(metadata_rows or {}) - steps
        if unmatched:
            step = min(unmatched)
            raise TrajectoryLoadError(
                sidecar,
                "metadata step has no trajectory row",
                row=(metadata_rows or {})[step][0],
                step=step,
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
            name: _decode_core_value(row[name], path, row_number, name) for name in cls._core_fields if name != "step"
        }
        extra = {name: value for name, value in row.items() if name not in cls._core_fields}

        for name in extra:
            domain = name.split(".", maxsplit=1)[0]
            if domain in cls._core_domains:
                raise TrajectoryLoadError(
                    path,
                    f"metadata overlaps core domain: {name}",
                    row=row_number,
                    step=step,
                )

        source_paths = {name: path for name in extra}

        if metadata_rows is not None:
            if step not in metadata_rows:
                raise TrajectoryLoadError(sidecar, "missing metadata row for trajectory step", step=step)
            metadata_row_number, metadata_row = metadata_rows[step]
            for name, value in metadata_row.items():
                if name != "step":
                    if name in extra:
                        raise TrajectoryLoadError(
                            sidecar,
                            f"duplicate metadata field: {name}",
                            row=metadata_row_number,
                            step=step,
                        )
                    extra[name] = value
                    source_paths[name] = sidecar

        for name, value in extra.items():
            _set_domain(domains, name, value, source_paths[name], step)

        return TrajectoryRecord(step, path, domains, row_number)

    def _record_at(self, index: int) -> TrajectoryRecord:
        row = self._rows[index]
        record = self._records[index]
        existing = {} if record is None else record._domains
        missing = {
            name: value
            for name, value in row.items()
            if name != "step" and name.split(".", maxsplit=1)[0] not in existing
        }
        if record is None or missing:
            domains = dict(existing)
            for name, value in missing.items():
                _set_domain(domains, name, value, self.output_path, row["step"])
            source = self.output_path if record is None else record.source
            source_row = None if record is None else record.row
            record = TrajectoryRecord(row["step"], source, domains, source_row)
            self._records[index] = record
        return record

    def records(self) -> tuple[TrajectoryRecord, ...]:
        return deepcopy(tuple(self._record_at(index) for index in range(len(self._rows))))

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
        return self._iteration_dir / self.file_name

    @property
    def metadata_path(self) -> Path:
        """Return the trajectory metadata CSV path."""
        return self._iteration_dir / self.metadata_file_name

    def append(
        self,
        *,
        step: int,
        action: object,
        reward: object,
        observation: object,
        **values: object,
    ) -> pd.Series:
        if self._writer_finalizer is None or not self._writer_finalizer.alive:
            raise RuntimeError("loaded trajectories are read-only; open an explicit resume or adopt writer")
        self._validate_writer_paths()
        record, snapshot, core_row, metadata_row = self._prepare_record(step, action, reward, observation, values)
        core_header, core_bytes = _encode_csv_row(core_row, self._headers.get("trajectory"))
        metadata_header, metadata_bytes = _encode_csv_row(metadata_row, self._headers.get("metadata"))
        record_digest = hashlib.sha256(core_bytes + b"\x00" + metadata_bytes).hexdigest()
        legacy_digest = _legacy_record_digest(
            tuple(core_row[name] for name in core_header), tuple(metadata_row[name] for name in metadata_header)
        )
        if self._reconcile_retry(step, (record_digest, legacy_digest)):
            return _copy_series(lazy.pd.Series(record, dtype=object))
        if self._rows and step <= self._rows[-1]["step"]:
            raise ValueError(f"trajectory steps must increase: last step is {self._rows[-1]['step']}, got {step}")
        self._recover_tails()
        publication, payloads, hashes = self._prepare_publication(
            step, record_digest, (core_header, core_bytes), (metadata_header, metadata_bytes)
        )
        self._pending = {
            "publication": publication,
            "hashes": hashes,
            "headers": {"trajectory": core_header, "metadata": metadata_header},
            "record": record,
            "snapshot": snapshot,
        }
        _append_synced(self.output_path, payloads[0])
        _append_synced(self.metadata_path, payloads[1])
        if self._publication is not None and not self._publication["trajectory"]["bytes"]:
            _sync_directory(self._iteration_dir)
        _publish_commit(self._iteration_dir, publication)
        self._accept_pending()
        logging.debug("Appended trajectory row for step %s (total rows: %s).", step, len(self))
        return _copy_series(lazy.pd.Series(record, dtype=object))

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

        # One index per distinct criteria field-set, built on first use. Callers query with a
        # stable set of domains, so this is built once and maintained by ``append``.
        fields = tuple(sorted(criteria))
        index = self._find_index.get(fields)
        if index is None:
            index = {}
            for position, record in enumerate(self._rows):
                if all(field in record for field in fields):
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
            self._validate_step(step)
            if previous_step is not None and step <= previous_step:
                raise ValueError(f"trajectory steps must increase: last step is {previous_step}, got {step}")
            previous_step = step

    @staticmethod
    def _validate_step(step: object) -> None:
        if type(step) is not int or step < 1:
            raise ValueError(f"trajectory step must be a positive integer; got {step}")

    def _initialize_state(self, iteration_dir: Path, dataframe: pd.DataFrame | None = None) -> None:
        self._iteration_dir = Path(iteration_dir)
        self._dataframe = lazy.pd.DataFrame() if dataframe is None else _copy_dataframe(dataframe)
        self._validate_dataframe()
        self._dataframe = self._dataframe.astype(object)
        self._rows: list[dict[str, Any]] = (
            [dict(record) for _, record in self._dataframe.iterrows()] if len(self._dataframe) else []
        )
        self._find_index: dict[tuple[str, ...], dict[tuple, list[int]]] = {}
        self._frame_dirty = False

        self._records: list[TrajectoryRecord | None] = [None] * len(self._rows)
        self._source_path: Path | None = None

        logging.debug(
            "Initializing Trajectory: entries=%s, columns=%s.",
            len(self),
            list(self._dataframe.columns),
        )

        self._writer_finalizer: weakref.finalize | None = None
        self._publication: dict[str, Any] | None = None
        self._pending: dict[str, Any] | None = None
        self._headers: dict[str, tuple[str, ...]] = {}
        self._hashes: dict[str, Any] = {}

    @classmethod
    def _load_snapshot(cls, path: Path, marker: dict[str, Any] | None, missing_ok: bool) -> Trajectory:
        sidecar = path.parent / cls.metadata_file_name
        texts = _committed_texts(path, marker) if marker is not None else None
        if (marker is not None and marker["trajectory"]["bytes"] == 0) or (
            marker is None
            and missing_ok
            and not any(candidate.exists() or candidate.is_symlink() for candidate in (path, sidecar))
        ):
            loaded = cls.__new__(cls)
            loaded._initialize_state(path.parent)
            loaded._source_path = path
            loaded._publication = marker
            return loaded
        columns, rows = _read_csv(path, required=cls._core_fields, text=None if texts is None else texts[0])
        has_metadata = texts is not None or sidecar.exists() or sidecar.is_symlink()
        if has_metadata:
            metadata_columns, metadata_rows = cls._read_metadata(sidecar, text=None if texts is None else texts[1])
            raw_metadata = list(metadata_rows.values())
        else:
            metadata_columns = []
            metadata_rows = None
            raw_metadata = []
        records, flattened_rows = cls._load_records(path, rows, sidecar, metadata_rows)
        frame = lazy.pd.DataFrame(flattened_rows, dtype=object)
        if not flattened_rows:
            frame = lazy.pd.DataFrame(columns=columns)
        loaded = cls.__new__(cls)
        loaded._initialize_state(path.parent)
        loaded._dataframe = _copy_dataframe(frame)
        loaded._rows = deepcopy(flattened_rows)
        loaded._source_path = path
        loaded._records = list(records)
        loaded._headers["trajectory"] = tuple(columns)
        if has_metadata:
            loaded._headers["metadata"] = tuple(metadata_columns)
        if marker is not None:
            _validate_committed_records(marker, rows, raw_metadata, path)
        loaded._publication = marker
        return loaded

    def _prepare_record(
        self, step: int, action: object, reward: object, observation: object, values: dict[str, object]
    ) -> tuple[dict[str, Any], TrajectoryRecord, dict[str, object], dict[str, object]]:
        self._validate_step(step)
        record: dict[str, Any] = {"step": step}
        for domain, value in {"action": action, "reward": reward, "observation": observation, **values}.items():
            flattened = _flatten_value(domain, value)
            duplicate = record.keys() & flattened.keys()
            if duplicate:
                raise ValueError(f"trajectory values produce duplicate column: {min(duplicate)}")
            record.update(deepcopy(flattened))
        if any(name.split(".", maxsplit=1)[0] in self._core_domains for name in values):
            raise ValueError("metadata overlaps a core trajectory domain")
        core_row = {
            "step": step,
            "action": action,
            "reward": reward,
            "observation": list(observation.values()) if isinstance(observation, Mapping) else observation,
        }
        metadata_row = {
            name: value
            for name, value in record.items()
            if name == "step" or name.split(".", maxsplit=1)[0] not in self._core_domains
        }
        serialized_core = {name: _csv_cell(value) for name, value in core_row.items()}
        serialized_metadata = {name: _csv_cell(value) for name, value in metadata_row.items()}
        self._build_record(
            self.output_path,
            2,
            serialized_core,
            step=step,
            sidecar=self.metadata_path,
            metadata_rows={step: (2, serialized_metadata)},
        )
        snapshot = TrajectoryRecord(
            step, self.output_path, {"action": action, "reward": reward, "observation": observation}
        )
        return record, snapshot, core_row, metadata_row

    def _validate_writer_paths(self) -> None:
        for path in (self.output_path, self.metadata_path):
            try:
                state = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
                raise TrajectoryLoadError(path, "writer data paths must be regular, nonsymlink, single-link files")

    def _open_writer(self, mode: str) -> None:
        self._validate_writer_paths()
        marker = _read_commit(self._iteration_dir)
        if mode == "create":
            if marker is not None or any(
                path.is_symlink() or (path.exists() and path.stat().st_size)
                for path in (self.output_path, self.metadata_path)
            ):
                raise ValueError("history already exists; choose explicit resume or legacy adopt mode")
            marker = _empty_commit()
            _publish_commit(self._iteration_dir, marker)
        else:
            if (mode == "resume") != (marker is not None):
                raise ValueError("resume requires a commit marker; adopt requires a markerless legacy history")
            loaded = type(self).load(self._iteration_dir)
            columns = loaded._headers.get("trajectory")
            if columns is not None and set(columns) != set(self._core_fields):
                raise TrajectoryLoadError(
                    self.output_path,
                    f"unsupported writable trajectory columns: {columns!r}; use Trajectory.load for read-only access",
                )
            for name in ("_dataframe", "_rows", "_records", "_headers", "_source_path"):
                setattr(self, name, getattr(loaded, name))
            if marker is None:
                marker = self._adopt_legacy()
        self._publication = marker
        for name, path in (("trajectory", self.output_path), ("metadata", self.metadata_path)):
            data = _read_prefix(path, marker[name], marker["history_id"])
            self._hashes[name] = _history_digest(marker["history_id"], data)
        self._recover_tails()

    def _adopt_legacy(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            content = _csv_bytes(("step",)) + b"".join(_csv_bytes((row["step"],)) for row in self._rows)
            _replace_synced(self.metadata_path, content)
        core_columns, core_rows = _read_csv(self.output_path, required=self._core_fields)
        metadata_columns, metadata_rows = _read_csv(self.metadata_path, required=("step",))
        self._headers = {"trajectory": tuple(core_columns), "metadata": tuple(metadata_columns)}
        marker = _empty_commit()
        marker["generation"] = len(core_rows)
        marker["last_step"] = None if not core_rows else int(core_rows[-1][1]["step"])
        marker["last_record_sha256"] = _last_record_digest(core_rows, metadata_rows)
        for name, path in (("trajectory", self.output_path), ("metadata", self.metadata_path)):
            with path.open("rb") as handle:
                data = handle.read()
                os.fsync(handle.fileno())
            marker[name] = {"bytes": len(data), "sha256": _history_digest(marker["history_id"], data).hexdigest()}
        _sync_directory(self._iteration_dir)
        _publish_commit(self._iteration_dir, marker)
        return marker

    def _recover_tails(self) -> None:
        if self._publication is None:
            raise RuntimeError("trajectory has no publication boundary")
        for name, path in (("trajectory", self.output_path), ("metadata", self.metadata_path)):
            size = self._publication[name]["bytes"]
            if not path.exists():
                if size:
                    raise TrajectoryLoadError(path, "missing committed data")
                continue
            actual = path.stat().st_size
            if actual < size:
                raise TrajectoryLoadError(path, "committed prefix is shorter than its byte bound")
            if actual > size:
                with path.open("r+b") as handle:
                    handle.truncate(size)
                    handle.flush()
                    os.fsync(handle.fileno())

    def _prepare_publication(
        self,
        step: int,
        record_digest: str,
        core: tuple[tuple[str, ...], bytes],
        metadata: tuple[tuple[str, ...], bytes],
    ) -> tuple[dict[str, Any], tuple[bytes, bytes], dict[str, Any]]:
        if self._publication is None:
            raise RuntimeError("trajectory has no publication boundary")
        publication = deepcopy(self._publication)
        payloads = []
        hashes = {}
        for name, path, (header, row) in (
            ("trajectory", self.output_path, core),
            ("metadata", self.metadata_path, metadata),
        ):
            size = publication[name]["bytes"]
            prefix = _csv_bytes(header) if not size else _append_separator(path, size)
            payload = prefix + row
            digest = self._hashes[name].copy()
            digest.update(payload)
            publication[name] = {"bytes": size + len(payload), "sha256": digest.hexdigest()}
            payloads.append(payload)
            hashes[name] = digest
        publication.update(generation=publication["generation"] + 1, last_step=step, last_record_sha256=record_digest)
        return publication, (payloads[0], payloads[1]), hashes

    def _reconcile_retry(self, step: int, record_digests: tuple[str, str]) -> bool:
        marker = _read_commit(self._iteration_dir)
        if marker != self._publication:
            if self._pending is None or marker != self._pending["publication"]:
                raise TrajectoryLoadError(
                    self._iteration_dir / ".trajectory.commit.json", "publication changed unexpectedly"
                )
            type(self).load(self._iteration_dir)
            _sync_directory(self._iteration_dir)
            self._accept_pending()
        if marker is not None and step == marker["last_step"]:
            if marker["last_record_sha256"] not in record_digests:
                raise ValueError(
                    "trajectory steps must increase; retry payload differs from the committed record at the same step"
                )
            return True
        return False

    def _accept_pending(self) -> None:
        if self._pending is None:
            raise RuntimeError("trajectory has no pending publication")
        pending = self._pending
        record = pending["record"]
        self._publication = pending["publication"]
        self._hashes = pending["hashes"]
        self._headers = pending["headers"]
        self._rows.append(record)
        position = len(self._rows) - 1
        for fields, index in self._find_index.items():
            if all(field in record for field in fields):
                index.setdefault(tuple(_hashable_key(record[field]) for field in fields), []).append(position)
        self._frame_dirty = True
        self._records.append(pending["snapshot"])
        self._pending = None


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


def _read_csv(
    path: Path, *, required: Sequence[str], text: str | None = None
) -> tuple[list[str], list[tuple[int, dict[str, str]]]]:
    if text is None:
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                text = handle.read()
        except (OSError, UnicodeError) as error:
            raise TrajectoryLoadError(path, f"cannot read artifact: {error}") from error

    if "\x00" in text:
        raise TrajectoryLoadError(path, "artifact contains a NUL character")

    with Trajectory._csv_read_lock:
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

    return header, rows


def _read_step(value: str, path: Path, row: int) -> int:
    try:
        step = int(value)
    except ValueError as error:
        raise TrajectoryLoadError(path, f"step must be a nonnegative integer, got {value!r}", row=row) from error
    if step < 0:
        raise TrajectoryLoadError(
            path,
            f"step must be a nonnegative integer, got {value!r}",
            row=row,
            step=step,
        )
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


def _flatten_record(record: TrajectoryRecord) -> dict[str, Any]:
    flattened: dict[str, Any] = {"step": record.step}
    for name, value in record._domains.items():
        flattened.update(_flatten_value(name, value))
    return flattened


def _empty_commit() -> dict[str, Any]:
    history_id = str(uuid.uuid4())
    return {
        "format_version": 1,
        "history_id": history_id,
        "generation": 0,
        "last_step": None,
        "trajectory": {"bytes": 0, "sha256": _history_digest(history_id).hexdigest()},
        "metadata": {"bytes": 0, "sha256": _history_digest(history_id).hexdigest()},
        "last_record_sha256": None,
    }


def _read_commit(directory: Path) -> dict[str, Any] | None:
    path = directory / ".trajectory.commit.json"
    if not path.exists() and not path.is_symlink():
        return None
    try:
        marker = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
        _validate_commit(marker)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
        raise TrajectoryLoadError(path, f"invalid commit marker: {error}") from error
    return marker


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and not set(value).difference("0123456789abcdef")


def _validate_commit(marker: Any) -> None:
    expected = {
        "format_version",
        "history_id",
        "generation",
        "last_step",
        "trajectory",
        "metadata",
        "last_record_sha256",
    }
    if not isinstance(marker, dict) or marker.keys() != expected:
        raise ValueError("commit marker fields do not match the supported schema")
    if type(marker["format_version"]) is not int or marker["format_version"] != 1:
        raise ValueError("unsupported commit format_version")
    if not isinstance(marker["history_id"], str) or str(uuid.UUID(marker["history_id"])) != marker["history_id"]:
        raise ValueError("invalid history identity")
    for name in ("trajectory", "metadata"):
        prefix = marker[name]
        if not isinstance(prefix, dict) or prefix.keys() != {"bytes", "sha256"}:
            raise ValueError(f"invalid {name} prefix specification")
        if type(prefix["bytes"]) is not int or prefix["bytes"] < 0 or not _valid_digest(prefix["sha256"]):
            raise ValueError(f"invalid {name} byte bound or checksum")
    _validate_commit_progress(marker)


def _validate_commit_progress(marker: dict[str, Any]) -> None:
    generation, step = marker["generation"], marker["last_step"]
    if type(generation) is not int or generation < 0:
        raise ValueError("invalid commit generation")
    if generation == 0:
        if step is not None or marker["last_record_sha256"] is not None:
            raise ValueError("empty generation has a last record")
        if bool(marker["trajectory"]["bytes"]) != bool(marker["metadata"]["bytes"]):
            raise ValueError("empty generation has mismatched CSV boundaries")
    elif (
        type(step) is not int
        or step < 0
        or not _valid_digest(marker["last_record_sha256"])
        or not marker["trajectory"]["bytes"]
        or not marker["metadata"]["bytes"]
    ):
        raise ValueError("nonempty generation requires a valid final step and both committed CSVs")


def _read_prefix(path: Path, specification: dict[str, Any], history_id: str) -> bytes:
    size = specification["bytes"]
    try:
        if size:
            with path.open("rb") as handle:
                if os.fstat(handle.fileno()).st_size < size:
                    raise TrajectoryLoadError(path, "committed prefix is shorter than its byte bound")
                data = handle.read(size)
        else:
            data = b""
    except OSError as error:
        raise TrajectoryLoadError(path, f"cannot read committed prefix: {error}") from error
    if len(data) != size:
        raise TrajectoryLoadError(path, "committed prefix is shorter than its byte bound")
    if _history_digest(history_id, data).hexdigest() != specification["sha256"]:
        raise TrajectoryLoadError(path, "committed prefix checksum does not match its history marker")
    return data


def _committed_texts(path: Path, marker: dict[str, Any]) -> tuple[str, str]:
    texts = []
    for name, artifact in (("trajectory", path), ("metadata", path.parent / Trajectory.metadata_file_name)):
        data = _read_prefix(artifact, marker[name], marker["history_id"])
        try:
            texts.append(data.decode("utf-8"))
        except UnicodeError as error:
            raise TrajectoryLoadError(artifact, f"invalid committed text: {error}") from error
    return texts[0], texts[1]


def _validate_committed_records(
    marker: dict[str, Any],
    core_rows: list[tuple[int, dict[str, str]]],
    metadata_rows: list[tuple[int, dict[str, str]]],
    path: Path,
) -> None:
    if len(core_rows) != marker["generation"] or len(metadata_rows) != marker["generation"]:
        raise TrajectoryLoadError(path, "committed row counts do not match the marker generation")
    last_step = None if not core_rows else int(core_rows[-1][1]["step"])
    if last_step != marker["last_step"]:
        raise TrajectoryLoadError(path, "committed final step does not match the marker")
    digests = {_last_record_digest(core_rows, metadata_rows)}
    if core_rows:
        row = core_rows[-1][1]
        metadata = next(value for _, value in metadata_rows if int(value["step"]) == int(row["step"]))
        digests.add(_legacy_record_digest(tuple(row.values()), tuple(metadata.values())))
    if marker["last_record_sha256"] not in digests:
        raise TrajectoryLoadError(path, "committed final record does not match the marker")


def _last_record_digest(
    core_rows: list[tuple[int, dict[str, str]]], metadata_rows: list[tuple[int, dict[str, str]]]
) -> str | None:
    if not core_rows:
        return None
    row = core_rows[-1][1]
    metadata = next(value for _, value in metadata_rows if int(value["step"]) == int(row["step"]))
    return hashlib.sha256(_csv_bytes(tuple(row.values())) + b"\x00" + _csv_bytes(tuple(metadata.values()))).hexdigest()


def _legacy_record_digest(core: Sequence[object], metadata: Sequence[object]) -> str:
    rows = []
    for values in (core, metadata):
        cells = [_csv_cell(value) for value in values]
        encoded = [
            '"' + cell.replace('"', '""') + '"' if any(character in cell for character in ',"\n') else cell
            for cell in cells
        ]
        rows.append((",".join(encoded) + "\n").encode("utf-8"))
    return hashlib.sha256(rows[0] + b"\x00" + rows[1]).hexdigest()


def _csv_cell(value: object) -> str:
    return "" if value is None else str(value)


def _csv_bytes(values: Sequence[object]) -> bytes:
    cells = [_csv_cell(value) for value in values]
    if any("\x00" in value for value in cells):
        raise ValueError("trajectory CSV cells cannot contain NUL characters")
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\r\n").writerow(cells)
    return (output.getvalue().removesuffix("\r\n") + "\n").encode("utf-8")


def _encode_csv_row(row: dict[str, object], header: tuple[str, ...] | None) -> tuple[tuple[str, ...], bytes]:
    fields = tuple(row) if header is None else header
    if set(fields) != set(row):
        raise ValueError(
            f"trajectory record fields changed; file fields do not match: expected {fields}, got {tuple(row)}"
        )
    return fields, _csv_bytes(tuple(row[name] for name in fields))


def _append_separator(path: Path, size: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(size - 1)
        return b"" if handle.read(1) in (b"\n", b"\r") else b"\n"


def _append_synced(path: Path, data: bytes) -> None:
    with path.open("ab") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_durable_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    absolute = directory.absolute()
    for parent in reversed((absolute, *absolute.parents)):
        _sync_directory(parent)


def _replace_synced(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _publish_commit(directory: Path, marker: dict[str, Any]) -> None:
    _validate_commit(marker)
    _replace_synced(
        directory / ".trajectory.commit.json",
        (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON object key")
    return result


def _history_digest(history_id: str, data: bytes = b"") -> Any:
    digest = hashlib.sha256(history_id.encode("utf-8") + b"\x00")
    digest.update(data)
    return digest
