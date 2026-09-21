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

import datetime
import logging
import pathlib
import tempfile

import cloudai.metrics
import cloudai.models.output


def metric_output(observation: cloudai.metrics.MetricObservation) -> cloudai.models.output.Metric:
    """Convert a canonical observation to an output metric."""
    return cloudai.models.output.Metric(
        name=observation.metric.display_name,
        value=observation.value,
        unit=observation.metric.unit,
        dimensions=[
            cloudai.models.output.Dimension(name=cloudai.metrics.dimension_label(key), value=str(value))
            for key, value in sorted(observation.dimensions.items())
        ],
    )


class ExperimentOutput:
    """Collect experiment results and publish snapshots."""

    def __init__(self, experiment: cloudai.models.output.Experiment, output_path: pathlib.Path) -> None:
        self.experiment = experiment.model_copy(deep=True)
        self.output_path = output_path

    def update_run(self, test_id: str, run: cloudai.models.output.Run) -> None:
        """Store the latest state of a run, identified by test and output path."""
        test = next((test for test in self.experiment.tests if test.id == test_id), None)
        if test is None:
            raise KeyError(f"Unknown experiment test: {test_id}")
        recorded = run.model_copy(deep=True)
        for index, current in enumerate(test.runs):
            if current.path == run.path:
                test.runs[index] = recorded
                self._update_test_status(test)
                return
        test.runs.append(recorded)
        self._update_test_status(test)

    def update_test(self, test: cloudai.models.output.Test) -> None:
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

    def update_dse(
        self,
        test_id: str,
        space: dict[str, list[str | int | float]],
        candidates: list[tuple[int, dict[str, str | int | float]]],
    ) -> None:
        """Store the first ranked candidate with a successful run and its metrics."""
        test = next((test for test in self.experiment.tests if test.id == test_id), None)
        if test is None:
            raise KeyError(f"Unknown experiment test: {test_id}")
        completed_runs = {run.step: run for run in test.runs if run.status == "completed"}
        for step, config in candidates:
            if step in completed_runs:
                test.dse = cloudai.models.output.DSE(space=space, best_step=step, best_config=config)
                test.metrics = [metric.model_copy(deep=True) for metric in completed_runs[step].metrics]
                return
        test.dse = cloudai.models.output.DSE(space=space)
        test.metrics = []

    def snapshot(self) -> cloudai.models.output.Experiment:
        """Return an independent snapshot without finalizing the experiment."""
        full = self.experiment.model_copy(deep=True)
        self._update_timing(full)
        for test in full.tests:
            for run in test.runs:
                self._update_timing(run)
        return full

    def write(self) -> None:
        """Atomically replace experiment.json, warning on failure."""
        temporary_path: pathlib.Path | None = None
        try:
            content = self.snapshot().model_dump_json(indent=2)
            self.output_path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.output_path, prefix=".experiment.json.", delete=False
            ) as temporary:
                temporary_path = pathlib.Path(temporary.name)
                temporary.write(content + "\n")
            temporary_path.replace(self.output_path / "experiment.json")
        except Exception as exc:
            logging.warning("Cannot write experiment output: %s", exc)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as exc:
                    logging.warning("Cannot remove temporary experiment output %s: %s", temporary_path, exc)

    def finish(self, status: cloudai.models.output.Status, finish: datetime.datetime | None) -> None:
        for test in self.experiment.tests:
            for run in test.runs:
                if run.status in ("pending", "running"):
                    run.status = "unknown"
            self._update_test_status(test)
            self._update_test_metrics(test)
        if status == "completed":
            statuses = {test.status for test in self.experiment.tests}
            for outcome in ("failed", "cancelled", "unknown"):
                if outcome in statuses:
                    status = outcome
                    break
        for test in self.experiment.tests:
            if test.status in ("pending", "running"):
                test.status = "completed" if status == "completed" else "unknown"
        self.experiment.status = status
        self.experiment.finish = finish
        self._update_timing(self.experiment)
        self.write()

    @staticmethod
    def _update_test_metrics(test: cloudai.models.output.Test) -> None:
        if test.dse is not None:
            return
        test.metrics = []
        if len(test.runs) == 1:
            run = test.runs[0]
            if run.status == "completed" and run.step in (None, 0):
                test.metrics = [metric.model_copy(deep=True) for metric in run.metrics]

    @staticmethod
    def _update_test_status(test: cloudai.models.output.Test) -> None:
        statuses = {run.status for run in test.runs}
        if "failed" in statuses:
            test.status = "failed"
        elif "cancelled" in statuses:
            test.status = "cancelled"
        elif "running" in statuses:
            test.status = "running"
        elif "pending" in statuses:
            test.status = "pending"
        elif "unknown" in statuses:
            test.status = "unknown"
        elif statuses == {"completed"}:
            test.status = "completed"

    @staticmethod
    def _update_timing(record: cloudai.models.output.Experiment | cloudai.models.output.Run) -> None:
        for field in ("start", "finish"):
            value = getattr(record, field)
            if value is not None:
                setattr(
                    record, field, value.astimezone(datetime.timezone.utc) if value.utcoffset() is not None else None
                )
        end = record.finish
        if end is None and record.status == "running":
            end = datetime.datetime.now(datetime.timezone.utc)
        if record.start is not None and end is not None:
            record.duration = max((end - record.start).total_seconds(), 0.0)
