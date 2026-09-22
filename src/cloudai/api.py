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

from pathlib import Path

import cloudai.handlers
import cloudai.models.output


def run_experiment(
    scenario: str | Path,
    system: str | Path,
    wait: bool = True,
    *,
    tests_dir: Path | None = None,
    hook_dir: Path | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
    single_sbatch: bool = False,
    enable_cache_without_check: bool = False,
) -> cloudai.models.output.Experiment:
    """
    Run a scenario and return its experiment snapshot.

    Pass TOML content as str or configuration files as Path. Relative test paths
    require a scenario file. Optional directories have the same meaning as in
    the CLI. With wait=False, a separate process executes the experiment and
    returns an initial pending snapshot; use get_experiment to read later results.

    Configuration and setup errors raise exceptions. Workload failures are
    recorded in the returned experiment's status. This function does not invoke
    CLI logging configuration or install signal handlers.
    """
    return cloudai.handlers.run_experiment(
        scenario,
        system,
        wait,
        tests_dir=tests_dir,
        hook_dir=hook_dir,
        output_dir=output_dir,
        dry_run=dry_run,
        single_sbatch=single_sbatch,
        enable_cache_without_check=enable_cache_without_check,
    )


def list_experiments(system: str | Path) -> list[tuple[str, Path]]:
    """
    Return experiment IDs and local result directories under system.output_path.

    Pass system TOML as str or a configuration file as Path. A missing results
    directory returns an empty list. Unreadable or invalid experiment files raise
    exceptions rather than returning an incomplete catalog.
    """
    return cloudai.handlers.list_experiments(system)


def validate_scenario(
    scenario: str | Path,
    system: str | Path,
    *,
    tests_dir: Path | None = None,
    hook_dir: Path | None = None,
    single_sbatch: bool = False,
) -> tuple[bool, dict[str, str]]:
    """
    Validate TOML content or config files without installing or running jobs.

    Return (True, {}) on success, otherwise (False, errors), with configuration
    names as keys and error messages as values. This checks configuration and
    execution constraints, not whether cluster resources are currently available.
    """
    return cloudai.handlers.validate_scenario(
        scenario, system, tests_dir=tests_dir, hook_dir=hook_dir, single_sbatch=single_sbatch
    )


def get_experiment(exp: str | Path) -> cloudai.models.output.Experiment:
    """
    Read experiment.json from a result directory or a JSON file path.

    Both str and Path represent filesystem paths here. Missing files raise
    FileNotFoundError; invalid JSON or schema raises pydantic.ValidationError.
    This reads the saved snapshot without querying the scheduler.
    """
    return cloudai.handlers.get_experiment(exp)
