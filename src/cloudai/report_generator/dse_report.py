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

from __future__ import annotations

import ast
import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml

from cloudai.core import CommandGenStrategy, System, TestRun
from cloudai.models.scenario import TestRunDetails
from cloudai.systems.slurm import SlurmJobMetadata
from cloudai.util.lazy_imports import lazy

from .util import load_system_metadata

# https://gpus.io/en/gpus
# https://getdeploying.com/gpus
# https://docs.coreweave.com/platform/instances/gpu/
GPU_HOURLY_COST_USD = {
    "H100": 3.0,
    "B200": 5.5,
    "GB200": 11.00,
    "GB300": 8.0,
}


@dataclass(frozen=True)
class DSEParameterValue:
    """Represents DSE dimension value."""

    text: str
    is_best: bool


@dataclass(frozen=True)
class DSEParameterRow:
    """Represents a dimension in DSE."""

    name: str
    values: list[DSEParameterValue]


@dataclass(frozen=True)
class DSECaseIterationSummary:
    """Summary for DSE case iteration."""

    name: str
    saved_time: str
    saved_gpu_hours: str
    saved_usd: str
    gpu_label: str
    avg_step_runtime: str
    observed_runtime: str
    efficiency_ratio: str
    efficiency_steps: str
    best_config_toml: str
    parameter_rows: list[DSEParameterRow]
    reward_chart_data: dict[str, Any] | None


@dataclass(frozen=True)
class TrajectoryStep:
    """Enriched trajectory step for DSE."""

    step: int
    reward: float
    observation_text: str
    action: dict[str, Any]
    elapsed_time_sec: int | None
    is_successful: bool


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"

    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, sec = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if sec or not parts:
        parts.append(f"{sec}s")
    return " ".join(parts)


def format_float(value: float | None, precision: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def format_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _safe_literal_eval(raw: Any, default: Any) -> Any:
    if isinstance(raw, str):
        with contextlib.suppress(SyntaxError, ValueError):
            return ast.literal_eval(raw)
    return default


def _format_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _normalize_gpu_family(gpu_name: str | None) -> str | None:
    if not gpu_name:
        return None
    upper = gpu_name.upper()

    # sorted because of `B200 in GB200 is True`
    for family in sorted(GPU_HOURLY_COST_USD, key=len, reverse=True):
        if family in upper:
            return family
    return None


def _step_elapsed_time(step_dir: Path) -> int | None:
    slurm_job_path = step_dir / "slurm-job.toml"
    if not slurm_job_path.exists():
        return None

    with slurm_job_path.open() as f:
        try:
            metadata = SlurmJobMetadata.model_validate(toml.load(f))
        except Exception as exc:
            logging.debug(f"Error validating slurm job metadata for {slurm_job_path}: {exc}")
            return None

    return metadata.elapsed_time_sec


def calculate_saved_gpu_hours(
    system: System,
    total_runtime_sec: float,
    projected_runtime_sec: float,
    test_run_details: TestRunDetails,
) -> float | None:
    gpus_per_node = getattr(system, "gpus_per_node", None)
    total_gpu_hours = (
        (total_runtime_sec / 3600.0) * test_run_details.nnodes * gpus_per_node if gpus_per_node is not None else None
    )
    projected_gpu_hours = (
        (projected_runtime_sec / 3600.0) * test_run_details.nnodes * gpus_per_node
        if projected_runtime_sec is not None and gpus_per_node is not None
        else None
    )
    return (
        max(projected_gpu_hours - total_gpu_hours, 0.0)
        if projected_gpu_hours is not None and total_gpu_hours is not None
        else None
    )


def calculate_savings(saved_gpu_hours: float | None, gpu_arch_label: str | None) -> float | None:
    gpu_arch_family = _normalize_gpu_family(gpu_arch_label)
    return (
        saved_gpu_hours * GPU_HOURLY_COST_USD[gpu_arch_family]
        if saved_gpu_hours is not None and gpu_arch_family in GPU_HOURLY_COST_USD
        else None
    )


def get_best_step(steps: list[TrajectoryStep]) -> TrajectoryStep | None:
    successful_steps = [step for step in steps if step.is_successful]
    if not successful_steps:
        return None
    return max(successful_steps, key=lambda step: step.reward)


def _build_reward_chart_data(steps: list[TrajectoryStep]) -> dict[str, Any] | None:
    if not steps:
        return None

    best_step = get_best_step(steps)
    if best_step is None:
        return None

    return {
        "labels": [step.step for step in steps],
        "rewards": [step.reward for step in steps],
        "observations": [step.observation_text for step in steps],
        "best_index": best_step.step - 1,
    }


def _build_parameter_rows(param_space: dict[str, list[Any]], best_action: dict[str, Any]) -> list[DSEParameterRow]:
    rows: list[DSEParameterRow] = []
    for name, values in param_space.items():
        best_value = _format_scalar(best_action.get(name, "n/a"))
        rows.append(
            DSEParameterRow(
                name=name,
                values=[
                    DSEParameterValue(
                        text=_format_scalar(value),
                        is_best=_format_scalar(value) == best_value,
                    )
                    for value in values
                ],
            )
        )
    return rows


def _build_trajectory_steps(
    iteration_dir: Path,
    test_case: TestRun,
    test_runs: list[TestRun],
) -> list[TrajectoryStep] | None:
    trajectory_file = iteration_dir / "trajectory.csv"
    if not trajectory_file.is_file():
        logging.warning(f"No trajectory file found for {test_case.name} at {trajectory_file}")
        return None

    df = lazy.pd.read_csv(trajectory_file)
    if df.empty:
        logging.warning(f"No trajectory data found for {test_case.name} at {trajectory_file}")
        return None

    runs_by_step = {test_run.step: test_run for test_run in test_runs}
    steps: list[TrajectoryStep] = []
    for row in df.to_dict(orient="records"):
        step_no = int(row["step"])
        action = _safe_literal_eval(row.get("action"), {})
        if not isinstance(action, dict):
            action = {}
        observation = _safe_literal_eval(row.get("observation"), [])
        if not isinstance(observation, list):
            observation = [observation]
        step_run = runs_by_step.get(step_no)
        steps.append(
            TrajectoryStep(
                step=step_no,
                reward=float(row["reward"]),
                observation_text=", ".join(_format_scalar(value) for value in observation) if observation else "n/a",
                action=action,
                elapsed_time_sec=_step_elapsed_time(iteration_dir / str(step_no)),
                is_successful=step_run.test.was_run_successful(step_run).is_successful if step_run else False,
            )
        )

    if not steps:
        return None

    steps.sort(key=lambda step: step.step)
    return steps


def _build_iteration_summary(
    system: System,
    results_root: Path,
    test_case: TestRun,
    iteration: int,
    iteration_dir: Path,
    test_runs: list[TestRun],
) -> DSECaseIterationSummary | None:
    trajectory_steps = _build_trajectory_steps(iteration_dir, test_case, test_runs)
    if not trajectory_steps:
        return None

    best_step = get_best_step(trajectory_steps)
    if best_step is None:
        return None

    best_step_dump = iteration_dir / str(best_step.step) / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME
    if not best_step_dump.exists():
        logging.warning(f"No test run dump found for best DSE step at {best_step_dump}")
        return None

    with best_step_dump.open() as f:
        test_run_details = TestRunDetails.model_validate(toml.load(f))

    elapsed_times = [step.elapsed_time_sec for step in trajectory_steps if step.elapsed_time_sec is not None]
    if not elapsed_times:
        return None

    total_observed_runtime_sec = sum(elapsed_times)
    avg_step_duration_sec = total_observed_runtime_sec / len(elapsed_times)
    total_space = len(test_case.all_combinations)
    projected_runtime_sec = avg_step_duration_sec * total_space
    saved_runtime_sec = max(projected_runtime_sec - total_observed_runtime_sec, 0.0)

    metadata = load_system_metadata(iteration_dir / str(best_step.step), results_root)
    gpu_arch_label = metadata.system.gpu_arch_type if metadata else None
    saved_gpu_hours = calculate_saved_gpu_hours(
        system=system,
        total_runtime_sec=total_observed_runtime_sec,
        projected_runtime_sec=projected_runtime_sec,
        test_run_details=test_run_details,
    )
    estimated_saved_cost_usd = calculate_savings(saved_gpu_hours, gpu_arch_label)
    reduction_factor = total_space / len(trajectory_steps)

    return DSECaseIterationSummary(
        name=f"{test_case.name}-{iteration}",
        saved_time=format_duration(saved_runtime_sec),
        saved_gpu_hours=format_float(saved_gpu_hours, 2),
        saved_usd=format_money(estimated_saved_cost_usd),
        gpu_label=gpu_arch_label or "unknown",
        avg_step_runtime=format_duration(avg_step_duration_sec),
        observed_runtime=format_duration(total_observed_runtime_sec),
        efficiency_ratio=f"~{format_float(reduction_factor, 1)}x",
        efficiency_steps=f"{len(trajectory_steps):,} / {total_space:,} steps",
        best_config_toml=toml.dumps(test_run_details.test_definition.model_dump()),
        parameter_rows=_build_parameter_rows(test_case.param_space, best_step.action),
        reward_chart_data=_build_reward_chart_data(trajectory_steps),
    )


def build_dse_summaries(
    system: System,
    results_root: Path,
    loaded_test_runs: list[TestRun],
    test_cases: list[TestRun],
) -> list[DSECaseIterationSummary]:
    result: list[DSECaseIterationSummary] = []

    for test_case in test_cases:
        if not test_case.is_dse_job:
            continue

        case_root = results_root / test_case.name
        if not case_root.is_dir():
            continue

        for iteration in range(test_case.iterations):
            dse_iteration_runs = [
                tr for tr in loaded_test_runs if tr.name == test_case.name and tr.current_iteration == iteration
            ]

            iteration_dir = case_root / str(iteration)
            if not iteration_dir.is_dir():
                continue

            summary = _build_iteration_summary(
                system=system,
                results_root=results_root,
                test_case=test_case,
                iteration=iteration,
                iteration_dir=case_root / str(iteration),
                test_runs=dse_iteration_runs,
            )
            if summary is not None:
                result.append(summary)

    return result
