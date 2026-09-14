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

"""Experiment output interface skeleton; runner hooks are inactive until collector creation is implemented."""

from datetime import datetime
from pathlib import Path

from cloudai.models.output import Experiment, ExperimentShort, Run, Status, Test


class ExperimentOutput:
    """
    One collector per experiment, shared across iterations, DSE trials, and batch jobs.

    Runners provide normalized records, keeping scheduler-specific parsing outside this module.
    The methods below describe the intended interface and are deliberately unimplemented.
    """

    def __init__(self, experiment: Experiment, output_path: Path) -> None:
        self.experiment = experiment
        self.output_path = output_path

    def update_run(self, test_id: str, run: Run) -> None:
        """Upsert a logical run by test ID and run path before its mutable TestRun advances."""
        raise NotImplementedError

    def update_test(self, test: Test) -> None:
        """Update test metadata, summary metrics, and optional DSE results."""
        raise NotImplementedError

    def snapshot(self) -> tuple[ExperimentShort, Experiment]:
        """Derive consistent short and full snapshots without finalizing the experiment."""
        raise NotImplementedError

    def write(self) -> None:
        """Publish experiment-summary.json and experiment.json, atomically per file, warning on errors."""
        raise NotImplementedError

    def finish(self, status: Status, finish: datetime | None) -> None:
        """Finalize and publish after the whole experiment completes or fails."""
        raise NotImplementedError(f"Finalizing experiment output with status {status} at {finish} is not implemented")
