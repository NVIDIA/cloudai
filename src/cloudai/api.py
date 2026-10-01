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

import pathlib

import cloudai.handlers
import cloudai.models.output
from cloudai.core import Parser


def run_experiment(
    scenario: pathlib.Path,
    system: pathlib.Path,
    *,
    tests_dir: pathlib.Path | None = None,
    hook_dir: pathlib.Path | None = None,
) -> cloudai.models.output.Experiment:
    """
    Run a scenario synchronously and return its experiment snapshot.

    Configuration and setup errors raise exceptions. Workload failures are
    reflected in the returned experiment status.
    """
    parsed_system, tests, parsed_scenario = cloudai.handlers.load_experiment(
        scenario.expanduser().resolve(),
        system.expanduser().resolve(),
        tests_dir=tests_dir,
        hook_dir=hook_dir,
    )
    runner = cloudai.handlers.create_experiment_runner(parsed_system, parsed_scenario)
    cloudai.handlers.execute_experiment(runner, tests)
    return runner.runner.experiment_output.snapshot()


def list_experiments(system: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """List saved experiment IDs and result directories for a system."""
    output_dir = Parser.parse_system(system.expanduser().resolve()).output_path.expanduser().resolve()
    if not output_dir.exists():
        return []
    if not output_dir.is_dir():
        raise NotADirectoryError(output_dir)

    return [
        (experiment_file.parent.name, experiment_file.parent)
        for experiment_file in sorted(output_dir.glob("*/experiment.json"))
        if experiment_file.is_file()
    ]


def get_experiment(exp_id: str | pathlib.Path, system: str | pathlib.Path) -> cloudai.models.output.Experiment:
    """Load an experiment by ID or by its result directory or JSON path."""
    if isinstance(exp_id, pathlib.Path):
        path = exp_id.expanduser()
        if path.suffix != ".json":
            path /= "experiment.json"
    else:
        if exp_id in ("", ".", "..") or pathlib.Path(exp_id).name != exp_id:
            raise ValueError(f"Invalid experiment ID: {exp_id!r}")
        output_dir = Parser.parse_system(pathlib.Path(system).expanduser().resolve()).output_path.expanduser().resolve()
        path = output_dir / exp_id / "experiment.json"

    return cloudai.models.output.Experiment.model_validate_json(path.read_text(encoding="utf-8"))
