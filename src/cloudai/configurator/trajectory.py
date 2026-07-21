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

import csv
import logging
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import pandas as pd


class Trajectory:
    """An ordered DataFrame of DSE steps persisted to ``trajectory.csv``."""

    file_name = "trajectory.csv"

    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        *,
        iteration_dir: Path | Callable[[], Path] | None = None,
    ) -> None:
        self._iteration_dir = iteration_dir
        self._dataframe = lazy.pd.DataFrame() if dataframe is None else dataframe.copy(deep=True)
        self._validate_dataframe()
        self._dataframe = self._dataframe.astype(object)
        self._fields: tuple[str, ...] | None = (
            tuple(self._dataframe.columns) if len(self._dataframe.columns) > 0 else None
        )
        logging.debug(
            "Initializing Trajectory: entries=%s, columns=%s.",
            len(self),
            list(self._dataframe.columns),
        )

    def __len__(self) -> int:
        """Return the number of trajectory rows."""
        return len(self._dataframe)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return a copy of the trajectory DataFrame for analysis."""
        return self._dataframe.copy(deep=True)

    @property
    def output_path(self) -> Path | None:
        """Return the current trajectory CSV path when persistence is configured."""
        if self._iteration_dir is None:
            return None
        iteration_dir = self._iteration_dir() if callable(self._iteration_dir) else self._iteration_dir
        return iteration_dir / self.file_name

    def append(self, *, step: int, **values: object) -> pd.Series:
        """Flatten, persist, and store one trajectory row."""
        self._validate_step(step)
        if len(self) and step <= self._dataframe.iloc[-1]["step"]:
            raise ValueError(
                f"trajectory steps must increase: last step is {self._dataframe.iloc[-1]['step']}, got {step}"
            )

        record: dict[str, object] = {"step": step}
        for domain, value in values.items():
            _flatten_value(record, domain, value)

        fields = tuple(record)
        if self._fields is not None and fields != self._fields:
            raise ValueError(f"trajectory record fields changed: expected {self._fields}, got {fields}")

        row = lazy.pd.Series(record, dtype=object)
        self._persist(row, fields)

        row_frame = row.to_frame().T.astype(object)
        self._dataframe = lazy.pd.concat([self._dataframe, row_frame], ignore_index=True).astype(object)
        self._fields = fields
        logging.debug("Appended trajectory row for step %s (total rows: %s).", step, len(self))
        return row.copy(deep=True)

    def find(self, **values: object) -> pd.Series | None:
        """Return a copy of the first row matching all supplied domain values."""
        criteria: dict[str, object] = {}
        for domain, value in values.items():
            _flatten_value(criteria, domain, value)

        if any(field not in self._dataframe.columns for field in criteria):
            return None

        for _, row in self._dataframe.iterrows():
            if all(_values_match_exact(row[field], value) for field, value in criteria.items()):
                logging.debug("Found matching trajectory row at step %s for %s.", row["step"], values)
                return row.copy(deep=True)
        logging.debug("No matching trajectory row found for %s.", values)
        return None

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
        for step in self._dataframe["step"]:
            self._validate_step(step)
            if previous_step is not None and step <= previous_step:
                raise ValueError(f"trajectory steps must increase: last step is {previous_step}, got {step}")
            previous_step = int(step)

    @staticmethod
    def _validate_step(step: object) -> None:
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 1:
            raise ValueError(f"trajectory step must be a positive integer; got {step}")

    def _persist(self, row: pd.Series, fields: tuple[str, ...]) -> None:
        path = self.output_path
        if path is None:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        if not write_header:
            with path.open(newline="") as file:
                existing_fields = tuple(next(csv.reader(file), ()))
            if existing_fields != fields:
                raise ValueError(f"trajectory file fields do not match: expected {fields}, got {existing_fields}")

        row.to_frame().T.to_csv(path, mode="a", header=write_header, index=False)
        logging.debug("Wrote trajectory row to %s.", path)


def _flatten_value(record: dict[str, object], key: str, value: object) -> None:
    """Flatten mappings into dot-separated columns while preserving leaf values."""
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            if not isinstance(child_key, str):
                raise TypeError(f"trajectory mapping keys must be strings: {child_key}")
            _flatten_value(record, f"{key}.{child_key}", child_value)
        return
    if key in record:
        raise ValueError(f"trajectory values produce duplicate column: {key}")
    record[key] = value


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
