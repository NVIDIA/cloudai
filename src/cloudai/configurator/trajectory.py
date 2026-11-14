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
import contextlib
import csv
import hashlib
import io
import json
import logging
import math
import os
import stat
import tempfile
import uuid
import weakref
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

    from cloudai.core import RewardOverrides, System, TestRun
    from cloudai.models.output import Run


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

        if decode_numbers and self._domains.get("record_context_version") == "1":
            context = _decode_record_context(self._domains, self.source, self.step, field=key)
            if key in context:
                return context[key]
            return (
                _decode_environment(value)
                if key == "env_params" and self._domains.get("record_format_version") == "1"
                else value
            )
        if decode_numbers and key == "env_params" and self._domains.get("record_format_version") == "1":
            return _decode_environment(value)
        return _decode_numbers(value) if decode_numbers else value

    def format_observation(self) -> str:
        try:
            observation = self["observation"]
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

    def ineligibility_reasons(
        self,
        purpose: Literal["reference", "measurement", "decision", "learning"],
        *,
        current: tuple[TestRun, System, RewardOverrides] | None = None,
    ) -> tuple[str, ...]:
        if purpose not in {"reference", "measurement", "decision", "learning"}:
            raise ValueError(f"unsupported trajectory purpose: {purpose!r}")
        if purpose == "learning" and current is None:
            raise ValueError("learning eligibility requires current configuration")
        action = self["action"]
        reasons = []
        details = self.get("run_details", {}, decode_numbers=True)
        resolved = details.get("resolved_action", action) if isinstance(details, Mapping) else action
        if self.step == 0:
            reasons.append("historical_step_zero")
        if not isinstance(action, Mapping) or not _finite_record_value(action):
            reasons.append("action_unusable")
        if purpose != "reference" and (not isinstance(resolved, Mapping) or not _finite_record_value(resolved)):
            reasons.append("action_unusable")
        environment = self.get("env_params", {}, decode_numbers=True)
        if purpose != "reference" and (not isinstance(environment, Mapping) or not _finite_record_value(environment)):
            reasons.append("environment_incompatible")
        if purpose != "decision":
            reasons.extend(_record_outcome_reasons(self, reference=purpose == "reference"))
        if current is not None and purpose in {"decision", "learning"}:
            reasons.extend(_target_configuration_reasons(self, current, learning=purpose == "learning"))
        return tuple(dict.fromkeys(reasons))


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
    def load(
        cls, source: str | Path, *, missing_ok: bool = False, recovery: Literal["recorded"] | None = None
    ) -> Trajectory:
        if recovery not in (None, "recorded"):
            raise ValueError(f"unsupported trajectory recovery policy: {recovery!r}")
        source = Path(source)
        path = source / cls.file_name if source.is_dir() else source
        sidecar = path.parent / cls.metadata_file_name
        marker = _read_commit(path.parent)
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
        columns, rows, core_text = _read_csv(path, required=cls._core_fields, text=None if texts is None else texts[0])
        has_metadata = texts is not None or sidecar.exists() or sidecar.is_symlink()
        metadata_text = None if texts is None else texts[1]
        metadata_rows = cls._read_metadata(sidecar, text=metadata_text) if has_metadata else None
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
            metadata_columns, raw_metadata, sidecar_text = _read_csv(sidecar, required=("step",), text=metadata_text)
            loaded._headers["metadata"] = tuple(metadata_columns)
        else:
            raw_metadata = []
            sidecar_text = ""
        if marker is not None:
            _validate_committed_records(marker, rows, raw_metadata, path)
        loaded._publication = marker
        if recovery == "recorded" and any(record.get("record_context_version") != "1" for record in records):
            source_id = (
                marker["history_id"]
                if marker is not None
                else hashlib.sha256(core_text.encode("utf-8") + b"\x00" + sidecar_text.encode("utf-8")).hexdigest()
            )
            source_kind = "history_uuid" if marker is not None else "sha256"
            records = [
                record
                if record.get("record_context_version") == "1"
                else cls._recover_recorded_record(record, source_id, source_kind)
                for record in records
            ]
            loaded._records = list(records)
            loaded._rows = [cls._flatten_record(record) for record in records]
            loaded._dataframe = lazy.pd.DataFrame(loaded._rows, dtype=object)
        return loaded

    @classmethod
    def _read_metadata(cls, path: Path, *, text: str | None = None) -> dict[int, tuple[int, dict[str, str]]]:
        columns, rows, _ = _read_csv(path, required=("step",), text=text)
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
            flattened_rows.append(cls._flatten_record(record))
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
            name: _decode_core_value(row[name], path, row_number, name) for name in cls._core_fields if name != "step"
        }
        extra = {name: value for name, value in row.items() if name not in cls._core_fields}

        for name in extra:
            domain = name.split(".", maxsplit=1)[0]
            if domain in cls._core_domains:
                raise TrajectoryLoadError(path, f"metadata overlaps core domain: {name}", row=row_number, step=step)

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

        format_version = extra.get("record_format_version")
        if any(name.startswith("record_format_version.") for name in extra) or (
            format_version is not None and format_version != "1"
        ):
            raise TrajectoryLoadError(
                path, f"unsupported record_format_version or non-scalar version field: {format_version!r}", step=step
            )
        _decode_record_context(extra, path, step)
        for name, value in extra.items():
            _set_domain(domains, name, value, source_paths[name], step, format_version=format_version)

        return TrajectoryRecord(step, path, domains, row_number)

    @staticmethod
    def _flatten_record(record: TrajectoryRecord) -> dict[str, Any]:
        flattened: dict[str, Any] = {"step": record.step}
        for name, value in record._domains.items():
            if name == "env_params" and record.get("record_format_version") == "1":
                flattened.update(_flatten_environment(record.get("env_params", decode_numbers=True)))
            else:
                flattened.update(_flatten_value(name, value))
        return flattened

    @staticmethod
    def _recover_recorded_record(record: TrajectoryRecord, source_id: str, source_kind: str) -> TrajectoryRecord:
        import toml

        from cloudai.core import CommandGenStrategy

        relative = (
            Path(str(record.step)) / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME
            if record.step
            else Path(CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME)
        )
        dump_path = record.source.parent / relative
        try:
            values = record.metadata
            contexts = {
                name: deepcopy(values.get(name, {})) for name in ("run_details", "measurement", "evaluation_failure")
            }
            environment = record.get("env_params", {}, decode_numbers=True)
            if any(not isinstance(value, dict) for value in contexts.values()) or not isinstance(environment, dict):
                raise ValueError("scalar reserved context cannot be normalized")
            if "recovery" in contexts["run_details"]:
                raise ValueError("reserved recovery provenance collision")
            data = {}
            dump = {
                "relative_path": relative.as_posix(),
                "sha256": None,
                "status": "absent",
                "reason": "historical_dump_absent",
            }
            if dump_path.exists() or dump_path.is_symlink():
                content = dump_path.read_bytes()
                data = toml.loads(content.decode("utf-8"))
                if "step" in data and (type(data["step"]) is not int or data["step"] != record.step):
                    raise ValueError("historical dump step conflict or invalid integer identity")
                dump = {
                    "relative_path": relative.as_posix(),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "status": "matched" if "step" in data else "unverified_identity",
                    "reason": None if "step" in data else "historical_dump_step_missing",
                }
            evidence = data if dump["status"] == "matched" else {}
            configuration = evidence.get("test_definition", {})
            if not isinstance(configuration, dict):
                raise ValueError("historical test_definition must be a mapping")
            resources = {"NUM_NODES": evidence["nnodes"]} if "nnodes" in evidence else {}
            semantic_evidence = {
                name: evidence[name]
                for name in (
                    "test_definition",
                    "nnodes",
                    "nodes",
                    "name",
                    "current_iteration",
                    "output_path",
                    "exclude_nodes",
                    "pin_nodes",
                    "time_limit",
                    "extra_srun_args",
                )
                if name in evidence
            }
            semantic_evidence.update(
                {
                    name: configuration[name]
                    for name in ("agent_metrics", "agent_observation", "agent_reward_function")
                    if name in configuration
                }
            )
            _merge_recorded_values(values, semantic_evidence, values, record)
            action = record["action"] if isinstance(record["action"], Mapping) else {}
            resolved = _resolve_recorded_action(configuration, resources, action, missing_ok=True)
            if any(not _values_match_exact(action[name], value) for name, value in resolved.items()):
                raise ValueError("recorded action conflicts with historical configuration")
            supplied = _record_environment(deepcopy(configuration), resources, action)
            _validate_environment(supplied)
            original_env = values.get("env_params", {})
            original_paths = (
                {tuple(json.loads(name.split(".", 1)[1])) for name in _flatten_environment(original_env)}
                if "env_params" in values and isinstance(original_env, dict)
                else set()
            )
            supplied_paths = (
                {tuple(json.loads(name.split(".", 1)[1])) for name in _flatten_environment(supplied)}
                if supplied
                else set()
            )
            legacy = record.get("record_format_version") != "1"
            if legacy and any(
                ".".join(left) == ".".join(right) and left != right
                for left in original_paths
                for right in supplied_paths
            ):
                raise ValueError("ambiguous legacy dotted environment path conflicts with historical literal path")
            ambiguous = set(original_paths) if legacy else set()
            environment = _merge_recorded_values(
                environment,
                supplied,
                original_env if legacy else environment,
                record,
                restore_types=True,
                typed=not legacy,
                ambiguous=ambiguous,
            )
            details = {
                name: evidence[key]
                for name, key in {
                    "run_name": "name",
                    "iteration": "current_iteration",
                    "step": "step",
                    "output_path": "output_path",
                }.items()
                if key in evidence
            }
            launch = {
                name: evidence[key]
                for name, key in {
                    "num_nodes": "nnodes",
                    "nodes": "nodes",
                    "exclude_nodes": "exclude_nodes",
                    "pin_nodes": "pin_nodes",
                    "time_limit": "time_limit",
                    "extra_srun_args": "extra_srun_args",
                }.items()
                if key in evidence
            }
            details.update({"launch": launch} if launch else {})
            details.update(
                {
                    name: configuration[key]
                    for name, key in {
                        "agent_metrics": "agent_metrics",
                        "agent_observation": "agent_observation",
                        "reward_function": "agent_reward_function",
                    }.items()
                    if key in configuration
                }
            )
            details.update(
                {"evaluation_policy": evidence["evaluation_policy"]} if "evaluation_policy" in evidence else {}
            )
            details.update(
                {"agent_reward_function": configuration["agent_reward_function"]}
                if "agent_reward_function" in contexts["run_details"] and "agent_reward_function" in configuration
                else {}
            )
            run_details = _merge_recorded_values(contexts["run_details"], details, contexts["run_details"], record)
            measurement = {
                "kind": "unknown",
                "outcome": "unknown",
                "execution_status": "unknown",
                "origin": {"status": "unknown", "reason": "historical_measurement_origin_not_recorded"},
            } | contexts["measurement"]
            failure = (
                contexts["evaluation_failure"]
                if "evaluation_failure" in values
                else {"status": "unknown", "reason": "historical_failure_evidence_not_recorded"}
            )
            unknowns = {
                "action_space": "original_declaration_unverified",
                "measurement": "historical_measurement_evidence_incomplete",
            }
            unknowns.update(
                {
                    name: "historical_schema_field_not_recorded"
                    for name in ("agent_metrics", "agent_observation", "reward_function")
                    if name not in run_details
                }
            )
            unknowns.update(
                {"schema": "historical_observation_schema_incomplete"}
                if "agent_metrics" not in run_details or "agent_observation" not in run_details
                else {}
            )
            unknowns.update(
                {"evaluation_policy": "historical_actual_policy_not_recorded"}
                if "evaluation_policy" not in run_details
                else {}
            )
            unknowns.update(
                {"run_identity": "historical_identity_incomplete"}
                if any(name not in run_details for name in ("run_name", "iteration", "step", "output_path"))
                else {}
            )
            unknowns.update(
                {"env_params": "historical_environment_not_recorded"}
                if not environment
                else {"env_params": "historical_environment_completeness_unknown"}
            )
            recorded_launch = run_details.get("launch", {})
            unknowns.update(
                {
                    f"launch.{name}": "historical_launch_field_not_recorded"
                    for name in (
                        "num_nodes",
                        "nodes",
                        "exclude_nodes",
                        "pin_nodes",
                        "time_limit",
                        "extra_srun_args",
                        "system",
                    )
                    if not isinstance(recorded_launch, dict) or name not in recorded_launch
                }
            )
            run_details["recovery"] = {
                "policy": "recorded",
                "opaque_context_paths": [
                    json.loads(name.split(".", 1)[1])
                    for group in ("measurement", "evaluation_failure")
                    if group in values
                    for name in _flatten_environment(values[group], (group,))
                ],
                "source_id": source_id,
                "source_id_kind": source_kind,
                "source_step": record.step,
                "dump": dump,
                "historical_identity": {
                    name: data[name] for name in ("name", "current_iteration", "step", "output_path") if name in data
                },
                "context_completeness": {
                    "status": "unknown",
                    "reason": "historical_serialization_may_redact_or_omit_fields",
                },
                "unknowns": unknowns,
                "ambiguous_env_paths": [list(path) for path in sorted(ambiguous)],
                "recorded_env_paths": [list(path) for path in sorted(supplied_paths)],
            }
            domains = {
                name: deepcopy(value)
                for name, value in record._domains.items()
                if name
                not in {
                    "env_params",
                    "record_format_version",
                    "record_context_version",
                    "run_details",
                    "measurement",
                    "evaluation_failure",
                }
            }
            domains.update({"record_format_version": "1", "record_context_version": "1"})
            for name, value in _flatten_environment(environment).items():
                _set_domain(
                    domains,
                    name,
                    json.dumps(value, allow_nan=False, separators=(",", ":")),
                    record.source,
                    record.step,
                    format_version="1",
                )
            domains.update(
                {
                    name: json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
                    for name, value in {
                        "run_details": run_details,
                        "measurement": measurement,
                        "evaluation_failure": failure,
                    }.items()
                }
            )
            _decode_record_context(domains, record.source, record.step)
            return TrajectoryRecord(record.step, record.source, domains, record.row)
        except (OSError, UnicodeError, TypeError, ValueError) as error:
            raise TrajectoryLoadError(
                record.source, f"historical recovery failed at {relative.as_posix()}: {error}", step=record.step
            ) from error

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
                format_version = row.get("record_format_version")
                if format_version == "1" and name.startswith("env_params."):
                    value = json.dumps(value, allow_nan=False, separators=(",", ":"))
                _set_domain(domains, name, value, self.output_path, row["step"], format_version=format_version)
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
        return self._source_path if self._source_path is not None else self._iteration_dir / self.file_name

    @property
    def metadata_path(self) -> Path:
        """Return the trajectory metadata CSV path."""
        return self._iteration_dir / self.metadata_file_name

    def record(
        self,
        *,
        test_run: TestRun,
        action_space: Mapping[str, Any],
        action: object,
        reward: object,
        observation: object,
        system: System,
        rewards: RewardOverrides,
        failed_metrics: Sequence[str] = (),
        execution_error: str | None = None,
        cached_result: pd.Series | None = None,
        run_output: Run | None = None,
    ) -> TrajectoryRecord:
        cached_measurement = (
            _decode_record_context(dict(cached_result), self.output_path, test_run.step, field="measurement")[
                "measurement"
            ]
            if cached_result is not None
            else {}
        )
        units = (
            {metric.name: metric.unit or "unknown" for metric in run_output.metrics} if run_output is not None else {}
        )
        if cached_result is None:
            with contextlib.suppress(Exception):
                units.update(
                    {
                        entry.metric.key: entry.metric.unit or "unknown"
                        for entry in test_run.test.metric_observations(system, test_run)
                        if units.get(entry.metric.key, "unknown") == "unknown"
                    }
                )
        observation_values = (
            observation
            if isinstance(observation, Mapping)
            else dict(zip(test_run.test.agent_metrics, observation, strict=False))
            if isinstance(observation, Sequence)
            else {}
        )
        metrics = {}
        for index, name in enumerate(test_run.test.agent_metrics):
            value = observation_values.get(name)
            status = "failed" if name in failed_metrics else "observed"
            if isinstance(value, float) and not math.isfinite(value):
                status = "nonfinite"
            unit = units.get(name, "unknown")
            if cached_result is not None:
                cached_metric = cached_measurement.get("metrics", {}).get(str(index), {})
                status = cached_metric.get("status", "unknown")
                unit = cached_metric.get("unit", "unknown")
            metrics[str(index)] = {"name": name, "unit": unit, "status": status}
        execution_status = run_output.status if run_output is not None else "unknown"
        outcome = "success" if execution_status == "completed" else "unknown"
        if (isinstance(reward, float) and not math.isfinite(reward)) or any(
            metric["status"] == "nonfinite" for metric in metrics.values()
        ):
            outcome = "nonfinite"
        if failed_metrics:
            outcome = "metric_failure"
        if execution_error is not None or execution_status in {"failed", "cancelled"}:
            outcome = "runner_failure"
        origin = {
            "history_id": self._publication["history_id"] if self._publication is not None else "unknown",
            "run_name": test_run.name,
            "iteration": test_run.current_iteration,
            "step": test_run.step,
            "output_path": str(test_run.output_path),
            "trajectory": str(self.output_path),
            "jobid": run_output.jobid if run_output is not None else "unknown",
        }
        if cached_result is not None:
            origin = {name: cached_measurement.get("origin", {}).get(name, "unknown") for name in origin}
            outcome = cached_measurement.get("outcome", "unknown")
            execution_error = cached_measurement.get("runner_error", "unknown")
            execution_status = cached_measurement.get("execution_status", "unknown")
        measurement = {
            "kind": "fresh" if cached_result is None else "cached",
            "outcome": outcome,
            "runner_error": execution_error or "",
            "execution_status": execution_status,
            "origin": origin,
            "metrics": metrics,
        }
        context = self._record_context(
            test_run,
            action_space,
            action=action,
            system=system,
            rewards=rewards,
            measurement=measurement,
            history_id=self._publication["history_id"] if self._publication is not None else "unknown",
        )
        for name in ("run_details", "measurement", "evaluation_failure"):
            context[name] = json.dumps(context[name], sort_keys=True, allow_nan=False, separators=(",", ":"))
        context["record_context_version"] = "1"
        self.append(
            step=test_run.step,
            action=context.pop("action"),
            reward=reward,
            observation=observation,
            **context,
        )
        return deepcopy(self._record_at(len(self._rows) - 1))

    @staticmethod
    def _record_context(
        test_run: TestRun | None,
        action_space: Mapping[str, Any] | None,
        *,
        action: object,
        system: System | None,
        rewards: RewardOverrides | None,
        measurement: dict[str, Any],
        history_id: str,
    ) -> dict[str, Any]:
        from pydantic import TypeAdapter

        if test_run is None or action_space is None or system is None or rewards is None:
            raise ValueError("test_run, action_space, system and rewards must be supplied together")

        environment = _record_environment(
            test_run.test.model_dump(mode="python"), {"NUM_NODES": test_run.num_nodes}, action_space
        )
        _validate_environment(environment)
        launch_fields = {
            "name",
            "scheduler",
            "global_env_vars",
            "default_partition",
            "account",
            "distribution",
            "mpi",
            "gpus_per_node",
            "ntasks_per_node",
            "extra_srun_args",
            "extra_sbatch_args",
            "container_mount_home",
            "default_namespace",
            "use_host_network",
            "project_name",
            "default_queue",
            "app",
            "os_version",
            "project_id",
            "cluster_id",
        }
        backend = system.model_dump(mode="python", include=launch_fields)
        backend["type"] = f"{type(system).__module__}.{type(system).__qualname__}"
        launch = {
            "num_nodes": test_run.num_nodes,
            "nodes": test_run.nodes,
            "exclude_nodes": test_run.exclude_nodes,
            "pin_nodes": test_run.pin_nodes,
            "time_limit": test_run.time_limit,
            "extra_srun_args": test_run.extra_srun_args,
            "system": backend,
        }
        record = {
            "action": action,
            "run_details": {
                "history_id": history_id,
                "run_name": test_run.name,
                "iteration": test_run.current_iteration,
                "step": test_run.step,
                "output_path": str(test_run.output_path),
                "agent_metrics": test_run.test.agent_metrics,
                "reward_function": test_run.test.agent_reward_function,
                "evaluation_policy": {
                    name: {"nonfinite": repr(value)} if isinstance(value, float) and not math.isfinite(value) else value
                    for name, value in rewards.model_dump(mode="python").items()
                },
                "launch": launch,
            },
            "measurement": measurement,
            "evaluation_failure": (
                {
                    "constraint": False,
                    "metric_indices": [
                        int(index)
                        for index, metric in measurement.get("metrics", {}).items()
                        if metric["status"] in {"failed", "nonfinite"}
                    ],
                    "runner": measurement.get("outcome") == "runner_failure",
                }
                if measurement.get("outcome") in {"runner_failure", "metric_failure", "nonfinite"}
                else None
            ),
        }
        _validate_environment(record)
        recording_context = {
            "purpose": "trajectory",
            "record": TypeAdapter(dict[str, Any]).dump_python(record, mode="json"),
        }
        configuration = test_run.test.model_dump(mode="json", context=recording_context)
        record = recording_context["record"]
        _validate_environment(record)
        resolved_action = _resolve_recorded_action(
            configuration, {"NUM_NODES": record["run_details"]["launch"]["num_nodes"]}, action_space
        )
        record["run_details"]["resolved_action"] = resolved_action
        environment = _record_environment(configuration, {"NUM_NODES": test_run.num_nodes}, action_space)
        _validate_environment(environment)
        identity = {
            "action": resolved_action,
            "env_params": environment,
            "launch": record["run_details"]["launch"],
            "agent_metrics": record["run_details"]["agent_metrics"],
            "reward_function": record["run_details"]["reward_function"],
            "evaluation_policy": record["run_details"]["evaluation_policy"],
        }
        encoded = json.dumps(identity, sort_keys=True, allow_nan=False, separators=(",", ":"))
        return {
            **record,
            "env_params": environment,
            "configuration_id": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "record_format_version": "1",
        }

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
        if self._reconcile_retry(step, record_digest):
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

    def _prepare_record(
        self, step: int, action: object, reward: object, observation: object, values: dict[str, object]
    ) -> tuple[dict[str, Any], TrajectoryRecord, dict[str, object], dict[str, object]]:
        _validate_step(step)
        format_version = values.get("record_format_version")
        if format_version is not None and format_version != "1":
            raise ValueError(f"unsupported record_format_version: {format_version!r}")
        context_version = values.get("record_context_version")
        if context_version is not None and context_version != "1":
            raise ValueError(f"unsupported record_context_version: {context_version!r}")
        record: dict[str, Any] = {"step": step}
        for domain, value in {"action": action, "reward": reward, "observation": observation, **values}.items():
            flattened = (
                _flatten_environment(value)
                if domain == "env_params" and format_version == "1"
                else _flatten_value(domain, value)
            )
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
        metadata_record = dict(record)
        if format_version == "1" and "env_params" in values:
            metadata_record = {name: value for name, value in record.items() if not name.startswith("env_params.")}
            metadata_record.update(_flatten_environment(values["env_params"], expand_mappings=False))
            header = self._headers.get("metadata")
            if header is not None and {
                name for name in metadata_record if name == "step" or name.split(".", 1)[0] not in self._core_domains
            } != set(header):
                metadata_record = record
        metadata = _serialize_metadata(lazy.pd.Series(metadata_record, dtype=object), self.metadata_path)
        metadata_row: dict[str, object] = (
            {"step": step} if metadata is None else {str(name): value for name, value in metadata.items()}
        )
        serialized_core = {name: _csv_cell(value) for name, value in core_row.items()}
        serialized_metadata = {name: _csv_cell(value) for name, value in metadata_row.items()}
        validated = self._build_record(
            self.output_path,
            2,
            serialized_core,
            step=step,
            sidecar=self.metadata_path,
            metadata_rows={step: (2, serialized_metadata)},
        )
        snapshot = TrajectoryRecord(
            step,
            self.output_path,
            (
                {"env_params": validated._domains["env_params"]}
                if format_version == "1" and "env_params" in validated._domains
                else {}
            )
            | {"action": action, "reward": reward, "observation": observation},
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
        core_columns, core_rows, _ = _read_csv(self.output_path, required=self._core_fields)
        metadata_columns, metadata_rows, _ = _read_csv(self.metadata_path, required=("step",))
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

    def _reconcile_retry(self, step: int, record_digest: str) -> bool:
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
            if record_digest != marker["last_record_sha256"]:
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

    def find(
        self,
        test_run: TestRun | None = None,
        action_space: Mapping[str, Any] | None = None,
        system: System | None = None,
        rewards: RewardOverrides | None = None,
        /,
        **values: object,
    ) -> pd.Series | None:
        if any(value is not None for value in (test_run, action_space, system, rewards)):
            context = self._record_context(
                test_run,
                action_space,
                action=values.get("action"),
                system=system,
                rewards=rewards,
                measurement={},
                history_id=self._publication["history_id"] if self._publication is not None else "unknown",
            )
            values = {name: value for name, value in values.items() if name not in {"action", "env_params"}} | {
                "configuration_id": context["configuration_id"],
                "record_context_version": "1",
                "record_format_version": "1",
            }
        criteria: dict[str, object] = {}
        for domain, value in values.items():
            if domain == "env_params":
                continue
            flattened = _flatten_value(domain, value)
            duplicate = criteria.keys() & flattened.keys()
            if duplicate:
                raise ValueError(f"trajectory values produce duplicate column: {min(duplicate)}")
            criteria.update(flattened)
        variants: list[tuple[dict[str, object], bool | None]] = [(criteria, None)]
        if "env_params" in values and values["env_params"] != {}:
            environment = values["env_params"]
            variants = []
            try:
                variants.append((criteria | _flatten_value("env_params", environment), False))
            except ValueError:
                if not isinstance(environment, Mapping):
                    raise
            if isinstance(environment, Mapping):
                variants.append((criteria | _flatten_environment(environment), True))
        positions = [
            position
            for variant, versioned in variants
            if (position := self._find_position(variant, versioned)) is not None
        ]
        if not positions:
            logging.debug("No matching trajectory row found for %s.", values)
            return None
        record = self._rows[min(positions)]
        logging.debug("Found matching trajectory row at step %s for %s.", record["step"], values)
        return _copy_series(lazy.pd.Series(record, dtype=object))

    def _find_position(self, criteria: dict[str, object], versioned: bool | None) -> int | None:
        fields = tuple(sorted(criteria))
        index = self._find_index.get(fields)
        if index is None:
            index = {}
            for position, record in enumerate(self._rows):
                if all(field in record for field in fields):
                    index.setdefault(tuple(_hashable_key(record[field]) for field in fields), []).append(position)
            self._find_index[fields] = index
        bucket = index.get(tuple(_hashable_key(criteria[field]) for field in fields), ())
        for position in bucket:
            candidate = self._rows[position]
            if versioned is not None and (candidate.get("record_format_version") == "1") != versioned:
                continue
            if all(_values_match_exact(candidate[field], criteria[field]) for field in fields):
                return position
        return None

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


def _read_csv(
    path: Path, *, required: Sequence[str], text: str | None = None
) -> tuple[list[str], list[tuple[int, dict[str, str]]], str]:
    if text is None:
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


def _set_domain(
    domains: dict[str, Any], name: str, value: Any, path: Path, step: int, *, format_version: Any = None
) -> None:
    if format_version == "1" and (name == "env_params" or name.startswith("env_params.")):
        parts, value = _decode_environment_field(name, value, path, step)
    else:
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


def _flatten_environment(value: object, path: tuple[str, ...] = (), *, expand_mappings: bool = True) -> dict[str, Any]:
    if not path and not isinstance(value, Mapping):
        raise TypeError("versioned env_params must be a mapping")
    if isinstance(value, Mapping) and value and (not path or expand_mappings):
        flattened = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"environment mapping keys must be strings: {key!r}")
            flattened.update(_flatten_environment(child, (*path, key), expand_mappings=expand_mappings))
        return flattened
    return {f"env_params.{json.dumps(path, separators=(',', ':'))}": deepcopy(value)}


def _decode_environment(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {name: _decode_environment(child) for name, child in value.items()}
    return json.loads(value)


def _serialize_metadata(row: pd.Series, path: Path) -> pd.Series | None:
    fields = [
        field
        for field in row.index
        if field == "step" or field.split(".", maxsplit=1)[0] not in Trajectory._core_domains
    ]
    if len(fields) == 1:
        return None
    metadata = row[fields].copy()
    if row.get("record_format_version") == "1":
        domains = {}
        for field in fields:
            if field == "step":
                continue
            if field.startswith("env_params."):
                _validate_environment(metadata[field])
                metadata[field] = json.dumps(metadata[field], allow_nan=False, separators=(",", ":"))
            _set_domain(domains, field, metadata[field], path, row["step"], format_version="1")
    return metadata


def _validate_environment(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"environment mapping keys must be strings: {key!r}")
            _validate_environment(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_environment(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("environment values must be finite")


def _decode_environment_field(name: str, value: Any, path: Path, step: int) -> tuple[list[str], Any]:

    try:
        environment_path = json.loads(name[len("env_params.") :])
        if not isinstance(environment_path, list) or any(not isinstance(part, str) for part in environment_path):
            raise ValueError("environment path must be a JSON array of strings")
        decoded = json.loads(value, object_pairs_hook=_unique_json_object)
        json.dumps(decoded, allow_nan=False)
        if not environment_path:
            if decoded != {}:
                raise ValueError("the root environment cell must contain an empty mapping")
            value = {}
    except (ValueError, TypeError) as error:
        raise TrajectoryLoadError(path, f"invalid versioned environment field {name!r}: {error}", step=step) from error
    return ["env_params", *environment_path], value


def _decode_record_context(
    values: Mapping[str, Any], path: Path, step: int, *, field: str | None = None
) -> dict[str, Any]:
    if any(name.startswith("record_context_version.") for name in values):
        raise TrajectoryLoadError(path, "record_context_version must be a scalar version", step=step)
    version = values.get("record_context_version")
    if version is None:
        return {}
    if version != "1":
        raise TrajectoryLoadError(path, f"unsupported record_context_version: {version!r}", step=step)
    required = {"run_details", "measurement", "evaluation_failure"}
    missing = required.difference(values)
    if missing:
        raise TrajectoryLoadError(path, f"missing record context cells: {sorted(missing)}", step=step)
    decoded = {}
    for name in required if field is None else required.intersection({field}):
        try:
            value = json.loads(values[name], object_pairs_hook=_unique_json_object)
            _validate_environment(value)
            if not isinstance(value, dict) and not (name == "evaluation_failure" and value is None):
                raise TypeError("context cell must contain an object or nullable evaluation_failure")
        except (TypeError, ValueError) as error:
            raise TrajectoryLoadError(path, f"invalid {name} context cell: {error}", step=step) from error
        decoded[name] = value
    return decoded


def _record_environment(
    configuration: dict[str, Any], resources: Mapping[str, Any], action_space: Mapping[str, Any]
) -> dict[str, Any]:
    environment = configuration.pop("cmd_args", {})
    if not isinstance(environment, dict):
        raise TypeError("recorded cmd_args must be a mapping")
    reserved = {"test_definition", "extra_env_vars", "extra_cmd_args"} | (
        {"NUM_NODES"} if "NUM_NODES" not in action_space else set()
    )
    conflicts = reserved.intersection(environment)
    if conflicts:
        raise ValueError(f"cmd_args contains reserved trajectory environment keys: {sorted(conflicts)}")
    for name in (name for name in action_space if name != "NUM_NODES" and not name.startswith("extra_env_vars.")):
        parts = name.split(".")
        target = environment
        for part in parts[:-1]:
            target = target.get(part, {}) if isinstance(target, dict) else {}
        if isinstance(target, dict):
            target.pop(parts[-1], None)
    extras = {name: configuration.pop(name) for name in ("extra_env_vars", "extra_cmd_args") if name in configuration}
    extra_environment = extras.get("extra_env_vars", {})
    if not isinstance(extra_environment, dict):
        raise TypeError("recorded extra_env_vars must be a mapping")
    for name in (name for name in action_space if name.startswith("extra_env_vars.")):
        extra_environment.pop(name[len("extra_env_vars.") :], None)
    environment.update(extras)
    environment.update(
        {"NUM_NODES": resources["NUM_NODES"]} if "NUM_NODES" not in action_space and "NUM_NODES" in resources else {}
    )
    controls = {
        "name",
        "description",
        "agent",
        "agent_steps",
        "agent_config",
        "agent_metrics",
        "agent_observation",
        "agent_reward_function",
        "training_report",
        "env_params",
        "dse_excluded_args",
    }
    configuration = {name: value for name, value in configuration.items() if name not in controls}
    environment.update({"test_definition": configuration} if configuration else {})
    return environment


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
    if _last_record_digest(core_rows, metadata_rows) != marker["last_record_sha256"]:
        raise TrajectoryLoadError(path, "committed final record does not match the marker")


def _last_record_digest(
    core_rows: list[tuple[int, dict[str, str]]], metadata_rows: list[tuple[int, dict[str, str]]]
) -> str | None:
    if not core_rows:
        return None
    row = core_rows[-1][1]
    metadata = next(value for _, value in metadata_rows if int(value["step"]) == int(row["step"]))
    return hashlib.sha256(_csv_bytes(tuple(row.values())) + b"\x00" + _csv_bytes(tuple(metadata.values()))).hexdigest()


def _csv_cell(value: object) -> str:
    return "" if value is None else str(value)


def _csv_bytes(values: Sequence[object]) -> bytes:
    cells = [_csv_cell(value) for value in values]
    if any("\x00" in value for value in cells):
        raise ValueError("trajectory CSV cells cannot contain NUL characters")
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerow(cells)
    return output.getvalue().encode("utf-8")


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


def _resolve_recorded_action(
    configuration: Mapping[str, Any],
    resources: Mapping[str, Any],
    action_space: Mapping[str, Any],
    *,
    missing_ok: bool = False,
) -> dict[str, Any]:
    resolved = {}
    for name in action_space:
        try:
            if name == "NUM_NODES":
                value = resources["NUM_NODES"]
            elif name.startswith("extra_env_vars."):
                value = configuration["extra_env_vars"][name[len("extra_env_vars.") :]]
            else:
                value = configuration["cmd_args"]
                if missing_ok and "." in name and name in value:
                    raise ValueError("ambiguous historical dotted action path")
                for part in name.split("."):
                    value = value[part]
        except (KeyError, TypeError):
            if missing_ok:
                continue
            raise
        resolved[name] = value
    return resolved


def _merge_recorded_values(
    existing: Mapping[str, Any],
    evidence: Mapping[str, Any],
    original: Mapping[str, Any],
    record: TrajectoryRecord,
    *,
    restore_types: bool = False,
    typed: bool = False,
    ambiguous: set[tuple[str, ...]] | None = None,
    prefix: tuple[str, ...] = (),
) -> dict[str, Any]:
    merged = deepcopy(dict(existing))
    for name, value in evidence.items():
        path = (*prefix, name)
        if name not in merged:
            merged[name] = deepcopy(value)
            continue
        prior = original[name]
        if isinstance(prior, Mapping) and isinstance(value, Mapping):
            merged[name] = _merge_recorded_values(
                merged[name],
                value,
                prior,
                record,
                restore_types=restore_types,
                typed=typed,
                ambiguous=ambiguous,
                prefix=path,
            )
            continue
        agrees = _values_match_exact(prior, value) if typed else _values_match_exact(prior, _csv_cell(value))
        if not agrees:
            raise TrajectoryLoadError(
                record.source, f"recorded context conflicts with historical value at {'.'.join(path)}", step=record.step
            )
        if restore_types:
            merged[name] = deepcopy(value)
            if ambiguous is not None:
                ambiguous.discard(path)
    return merged


def _finite_record_value(value: Any) -> bool:
    try:
        _validate_environment(value)
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _record_outcome_reasons(record: TrajectoryRecord, *, reference: bool) -> tuple[str, ...]:
    reasons = []
    try:
        reward = float(record["reward"])
        usable = not isinstance(record["reward"], bool) and math.isfinite(reward)
    except (TypeError, ValueError, OverflowError):
        reward, usable = None, False
    if not usable:
        reasons.append("reward_unusable")
    if reference:
        return tuple(reasons)
    if not _finite_record_value(record["observation"]):
        reasons.append("observation_unusable")
    failure = record.get("evaluation_failure", decode_numbers=True)
    measurement = record.get("measurement", {}, decode_numbers=True)
    measurement = measurement if isinstance(measurement, Mapping) else {}
    failure_fields = failure if isinstance(failure, Mapping) else {}
    details = record.get("run_details", {}, decode_numbers=True)
    recovery = details.get("recovery", {}) if isinstance(details, Mapping) else {}
    paths = recovery.get("opaque_context_paths", []) if isinstance(recovery, Mapping) else []
    opaque = (
        {tuple(path) for path in paths if isinstance(path, list) and all(isinstance(part, str) for part in path)}
        if isinstance(paths, list)
        else set()
    )
    failure_fields = {
        name: value
        for name, value in failure_fields.items()
        if not (
            ("evaluation_failure", name) in opaque
            and (
                (name in ("constraint", "runner", "metric", "nonfinite") and type(value) is not bool)
                or (
                    name == "metric_indices"
                    and (not isinstance(value, list) or not all(type(index) is int and index >= 0 for index in value))
                )
            )
        )
    }
    metrics = measurement.get("metrics", {})
    metrics = {} if ("measurement", "metrics") in opaque and not isinstance(metrics, Mapping) else metrics
    metrics = (
        {
            index: {
                name: value
                for name, value in metric.items()
                if not (
                    name in ("name", "unit", "status")
                    and ("measurement", "metrics", index, name) in opaque
                    and type(value) is not str
                )
            }
            if isinstance(metric, Mapping)
            else metric
            for index, metric in metrics.items()
            if not (("measurement", "metrics", index) in opaque and not isinstance(metric, Mapping))
        }
        if isinstance(metrics, Mapping)
        else metrics
    )
    indices = failure_fields.get("metric_indices")
    malformed = (
        not isinstance(metrics, Mapping)
        or any(
            type(measurement[name]) is not str
            for name in ("outcome", "execution_status", "kind")
            if name in measurement
        )
        or any(
            type(failure_fields[name]) is not bool
            for name in ("constraint", "runner", "metric", "nonfinite")
            if name in failure_fields
        )
        or (
            "metric_indices" in failure_fields
            and (not isinstance(indices, list) or not all(type(index) is int and index >= 0 for index in indices))
        )
        or any(
            not isinstance(metric, Mapping)
            or any(type(metric[name]) is not str for name in ("name", "unit", "status") if name in metric)
            for metric in (metrics.values() if isinstance(metrics, Mapping) else ())
        )
    )
    if record.get("record_context_version") == "1" and malformed:
        return (*reasons, "measurement_schema_incompatible")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    failed = (
        any(failure_fields.get(name) is True for name in ("constraint", "runner", "metric", "nonfinite"))
        or (isinstance(indices, list) and bool(indices) and all(type(index) is int and index >= 0 for index in indices))
        or measurement.get("outcome")
        in ("runner_failure", "metric_failure", "constraint_failure", "nonfinite", "failed")
        or measurement.get("execution_status") in ("failed", "cancelled")
        or any(
            isinstance(metric, Mapping) and metric.get("status") in ("failed", "nonfinite")
            for metric in metrics.values()
        )
    )
    if failed:
        reasons.append("explicit_failure")
    disambiguated = ("evaluation_failure" in record and failure is None) or measurement.get("outcome") == "success"
    if reward == -1 and not failed and not disambiguated:
        reasons.append("legacy_failure_ambiguous")
    return tuple(reasons)


def _target_configuration_reasons(
    record: TrajectoryRecord, current: tuple[TestRun, System, RewardOverrides], *, learning: bool
) -> tuple[str, ...]:
    from cloudai.configurator.env_params import EnvParams

    original, system, rewards = current
    details = record.get("run_details", {}, decode_numbers=True)
    required = {"resolved_action", "agent_metrics", "reward_function", "evaluation_policy", "launch"}
    if (
        record.get("record_context_version") != "1"
        or not isinstance(details, Mapping)
        or not required.issubset(details)
        or "recovery" in details
    ):
        return ("context_unavailable",)
    resolved = details["resolved_action"]
    proposal = record["action"]
    space = original.param_space
    if (
        not isinstance(resolved, dict)
        or not isinstance(proposal, Mapping)
        or set(resolved) != set(space)
        or any(
            name not in resolved or not _values_match_exact(value, resolved[name]) for name, value in proposal.items()
        )
        or any(
            not any(_values_match_exact(value, candidate) for candidate in space[name])
            for name, value in resolved.items()
        )
    ):
        return ("action_incompatible",)
    environment = record.get("env_params", {}, decode_numbers=True)
    parameters = EnvParams.from_test(original.test)
    sampled = {} if parameters is None else parameters.params
    if not isinstance(environment, Mapping) or set(sampled) != set(original.test.env_params):
        return ("environment_incompatible",)
    regime = {}
    for name, parameter in sampled.items():
        support = [
            value
            for index, value in enumerate(parameter.candidates)
            if parameter.weights is None or parameter.weights[index] > 0
        ]
        if name not in environment or not any(_values_match_exact(environment[name], value) for value in support):
            return ("environment_incompatible",)
        regime[name] = environment[name]
    try:
        submitted = original.apply_params_set(resolved, env_params=regime)
        expected = Trajectory._record_context(
            submitted, space, action=proposal, system=system, rewards=rewards, measurement={}, history_id="unknown"
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return ("configuration_incompatible",)
    expected_details = expected["run_details"]
    schema_matches = _values_match_exact(details["agent_metrics"], expected_details["agent_metrics"])
    measurement_matches = True
    if learning:
        measurement = record.get("measurement", {}, decode_numbers=True)
        metrics = measurement.get("metrics", {}) if isinstance(measurement, Mapping) else {}
        known_units = {}
        with contextlib.suppress(Exception):
            known_units = {
                item.metric.key: item.metric.unit
                for item in submitted.test.metric_observations(system, submitted)
                if item.metric.unit
            }
        observations = record["observation"]
        measurement_matches = (
            isinstance(metrics, Mapping)
            and isinstance(observations, list)
            and len(metrics) == len(expected_details["agent_metrics"])
            and len(observations) == len(expected_details["agent_metrics"])
            and all(type(value) in (int, float) and math.isfinite(value) for value in observations)
            and all(
                isinstance(metrics.get(str(index)), Mapping)
                and type(metrics[str(index)].get("name")) is str
                and metrics[str(index)].get("name") == name
                and metrics[str(index)].get("status") == "observed"
                and type(metrics[str(index)].get("unit")) is str
                and (name not in known_units or metrics[str(index)]["unit"] in ("unknown", known_units[name]))
                for index, name in enumerate(expected_details["agent_metrics"])
            )
        )
    return tuple(
        reason
        for reason, fails in (
            (
                "configuration_incompatible",
                not _values_match_exact(environment, expected["env_params"])
                or any(
                    not _values_match_exact(details[name], expected_details[name])
                    for name in ("launch", "reward_function", "evaluation_policy")
                ),
            ),
            ("measurement_schema_incompatible", not schema_matches or (learning and not measurement_matches)),
        )
        if fails
    )
