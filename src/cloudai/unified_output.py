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
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from tempfile import NamedTemporaryFile
from typing import Literal

import toml
from pydantic import BaseModel, Field, FiniteFloat

from cloudai.core import BaseJob, JobStatusResult, System, TestScenario
from cloudai.metrics import MetricCatalog, MetricObservation, MetricValue

Status = Literal["pending", "running", "completed", "failed", "cancelled", "unknown"]


class Dimension(BaseModel):
    """A metric coordinate in the unified output contract."""

    name: str
    value: str
    unit: str = ""
    is_x: bool = False


class Metric(BaseModel):
    """A canonical measurement and its dimension point."""

    name: str
    value: FiniteFloat
    unit: str
    dimensions: list[Dimension]


class Run(BaseModel):
    """One submitted execution, independent of the mutable TestRun."""

    path: str
    jobid: str
    status: Status = "unknown"
    metrics: list[Metric] = Field(default_factory=list)
    start: datetime | None = None
    finish: datetime | None = None
    duration: FiniteFloat | None = None
    iteration: int
    step: int


class TestResult(BaseModel):
    """A scenario test case and all its executed iterations."""

    id: str
    name: str
    description: str
    status: Status = "pending"
    path: str
    metrics: list[Metric] = Field(default_factory=list)
    runs: list[Run] = Field(default_factory=list)


class Experiment(BaseModel):
    """The API Schema v0.2 Experiment representation."""

    id: str
    name: str
    status: Status = "unknown"
    path: str
    start: datetime | None
    finish: datetime | None = None
    duration: FiniteFloat | None = None
    tests: list[TestResult]


def _timestamp(value: str) -> datetime | None:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    # sacct's default timestamps carry no timezone; the login host may use a different one.
    return timestamp.astimezone(timezone.utc) if timestamp.tzinfo is not None else None


def _metric(observation: MetricObservation) -> Metric:
    dimensions = []
    for key, value in sorted(observation.dimensions.items()):
        definition = MetricCatalog.get_dimension(key)
        dimensions.append(
            Dimension(name=definition.label, value=str(value), unit=definition.unit, is_x=definition.is_x)
        )
    return Metric(
        name=observation.metric.display_name,
        value=observation.value,
        unit=observation.metric.unit,
        dimensions=dimensions,
    )


def _status(statuses: set[Status]) -> Status:
    if "failed" in statuses:
        return "failed"
    if "cancelled" in statuses:
        return "cancelled"
    if statuses == {"pending"}:
        return "pending"
    return "completed" if statuses == {"completed"} else "unknown"


class ExperimentOutput:
    """Collect completed Slurm runs and atomically write one experiment.json."""

    def __init__(self, scenario: TestScenario, output_path: Path):
        self.scenario = scenario
        self.output_path = output_path.absolute()
        self.tests = {
            tr.name: TestResult(
                id=tr.name,
                name=tr.name,
                description=tr.test.description,
                path=str(self.output_path / tr.name),
            )
            for tr in scenario.test_runs
        }
        self.observations: dict[str, list[MetricObservation]] = defaultdict(list)
        self.experiment = Experiment(
            id=self.output_path.name,
            name=scenario.name,
            path=str(self.output_path),
            start=datetime.now(timezone.utc),
            tests=list(self.tests.values()),
        )

    def capture(self, system: System, job: BaseJob, result: JobStatusResult | None = None) -> None:
        tr = job.test_run
        test = self.tests[tr.name]
        if any(run.jobid == str(job.id) and run.path == str(tr.output_path.absolute()) for run in test.runs):
            return
        run = Run(
            path=str(tr.output_path.absolute()),
            jobid=str(job.id),
            iteration=tr.current_iteration,
            step=tr.step,
            status="failed" if result is not None and not result.is_successful else "unknown",
        )
        test.runs.append(run)
        metadata_path = tr.output_path / "slurm-job.toml"
        try:
            metadata = toml.load(metadata_path)
            state = metadata["state"].split()[0].rstrip("+")
            if state == "CANCELLED":
                run.status = "cancelled"
            elif state in {
                "FAILED",
                "TIMEOUT",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "BOOT_FAIL",
                "DEADLINE",
                "PREEMPTED",
                "REVOKED",
                "SPECIAL_EXIT",
            } or metadata["exit_code"] not in {"0", "0:0", ""}:
                run.status = "failed"
            elif state == "COMPLETED" and result is not None and result.is_successful:
                run.status = "completed"
            run.start = _timestamp(metadata["start_time"])
            run.finish = _timestamp(metadata["end_time"])
            run.duration = metadata["elapsed_time_sec"]
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            if result is not None or metadata_path.exists():
                logging.warning("Cannot read unified output metadata for job %s: %s", job.id, exc)

        if result is None:
            return
        try:
            observations = sorted(
                tr.test.metric_observations(system, tr),
                key=lambda observation: (observation.metric.key, sorted(observation.dimensions.items())),
            )
            run.metrics = [_metric(observation) for observation in observations]
            if run.status == "completed":
                self.observations[tr.name].extend(observations)
        except Exception as exc:
            logging.warning("Cannot extract unified output metrics for job %s: %s", job.id, exc)

    def finish(self, system: System, unfinished_jobs: list[BaseJob], completed: bool) -> None:
        temporary_path = None
        try:
            for job in unfinished_jobs:
                self.capture(system, job)
            for tr in self.scenario.test_runs:
                test = self.tests[tr.name]
                statuses: set[Status] = {run.status for run in test.runs}
                if len(test.runs) != tr.iterations:
                    statuses.add("unknown" if test.runs else "pending")
                test.status = _status(statuses)

                groups: dict[tuple[str, tuple[tuple[str, MetricValue], ...]], list[MetricObservation]] = defaultdict(
                    list
                )
                for observation in self.observations[tr.name]:
                    groups[(observation.metric.key, tuple(sorted(observation.dimensions.items())))].append(observation)
                test.metrics = [
                    _metric(replace(group[0], value=fmean(observation.value for observation in group)))
                    for _, group in sorted(groups.items())
                ]

            self.experiment.status = _status({test.status for test in self.tests.values()}) if completed else "failed"
            self.experiment.finish = datetime.now(timezone.utc)
            if self.experiment.start is not None:
                self.experiment.duration = (self.experiment.finish - self.experiment.start).total_seconds()
            content = self.experiment.model_dump_json(indent=2)
            self.output_path.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.output_path, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(content + "\n")
            temporary_path.replace(self.output_path / "experiment.json")
        except Exception as exc:
            logging.warning("Cannot write unified experiment output: %s", exc)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as exc:
                    logging.warning("Cannot remove temporary unified output %s: %s", temporary_path, exc)
