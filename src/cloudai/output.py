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

import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from tempfile import NamedTemporaryFile

from cloudai.models.output import Experiment, ExperimentShort, Metric, Run, Status, Test


class ExperimentOutput:
    """Collect experiment results and publish full and short snapshots."""

    def __init__(self, experiment: Experiment, output_path: Path) -> None:
        self.experiment = experiment.model_copy(deep=True)
        self.output_path = output_path

    def update_run(self, test_id: str, run: Run) -> None:
        """Store the latest state of a run, identified by test and output path."""
        test = next((test for test in self.experiment.tests if test.id == test_id), None)
        if test is None:
            raise KeyError(f"Unknown experiment test: {test_id}")
        recorded = run.model_copy(deep=True)
        for index, current in enumerate(test.runs):
            if current.path == run.path:
                test.runs[index] = recorded
                return
        test.runs.append(recorded)

    def update_test(self, test: Test) -> None:
        """Update a test while retaining its previously recorded runs."""
        recorded = test.model_copy(deep=True)
        for index, current in enumerate(self.experiment.tests):
            if current.id == test.id:
                incoming_runs = recorded.runs
                recorded.runs = current.runs
                self.experiment.tests[index] = recorded
                for run in incoming_runs:
                    self.update_run(test.id, run)
                return
        self.experiment.tests.append(recorded)

    def snapshot(self) -> tuple[ExperimentShort, Experiment]:
        """Return independent short and full views without finalizing the experiment."""
        full = self.experiment.model_copy(deep=True)
        self._update_timing(full)
        for test in full.tests:
            for run in test.runs:
                self._update_timing(run)
            if test.metrics or test.dse is not None or any(run.step not in (None, 0) for run in test.runs):
                continue
            test.metrics = self._aggregate_metrics(test.runs)
        short = ExperimentShort.model_validate(full.model_dump())
        return short, full

    def write(self) -> None:
        """Atomically replace each output file, warning on failure."""
        pending: list[tuple[Path, Path]] = []
        try:
            short, full = self.snapshot()
            contents = [
                ("experiment.json", full.model_dump_json(indent=2)),
                ("experiment-summary.json", short.model_dump_json(indent=2)),
            ]
            self.output_path.mkdir(parents=True, exist_ok=True)
            for filename, content in contents:
                with NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.output_path, prefix=f".{filename}.", delete=False
                ) as temporary:
                    pending.append((Path(temporary.name), self.output_path / filename))
                    temporary.write(content + "\n")
            for temporary_path, destination in pending:
                temporary_path.replace(destination)
        except Exception as exc:
            logging.warning("Cannot write experiment output: %s", exc)
        finally:
            for temporary_path, _ in pending:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as exc:
                    logging.warning("Cannot remove temporary experiment output %s: %s", temporary_path, exc)

    def finish(self, status: Status, finish: datetime | None) -> None:
        self.experiment.status = status
        self.experiment.finish = finish
        self._update_timing(self.experiment)
        self.write()

    @staticmethod
    def _update_timing(record: Experiment | Run) -> None:
        for field in ("start", "finish"):
            value = getattr(record, field)
            if value is not None:
                setattr(record, field, value.astimezone(timezone.utc) if value.utcoffset() is not None else None)
        end = record.finish
        if end is None and record.status == "running":
            end = datetime.now(timezone.utc)
        if record.start is not None and end is not None:
            record.duration = max((end - record.start).total_seconds(), 0.0)

    @staticmethod
    def _aggregate_metrics(runs: list[Run]) -> list[Metric]:
        groups: dict[tuple[str, str, tuple[tuple[str, str, str], ...]], list[Metric]] = defaultdict(list)
        for run in runs:
            if run.status != "completed":
                continue
            for metric in run.metrics:
                point = tuple(
                    sorted((dimension.name, dimension.value, dimension.unit) for dimension in metric.dimensions)
                )
                groups[(metric.name, metric.unit, point)].append(metric)

        metrics: list[Metric] = []
        for group in groups.values():
            metric = group[0].model_copy(deep=True)
            values = [item.value for item in group]
            if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                metric.value = fmean(float(value) for value in values)
            elif not all(type(value) is type(metric.value) and value == metric.value for value in values):
                logging.warning("Cannot aggregate conflicting values for metric %s", metric.name)
                continue
            metrics.append(metric)
        return metrics
