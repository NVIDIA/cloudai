# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import toml
from pydantic import BaseModel

from cloudai.core import CommandGenStrategy, TestRun, case_name
from cloudai.models.scenario import TestRunDetails
from cloudai.util.lazy_imports import lazy

GPU_HOURLY_COST_USD = {
    "H100": 4.50,
    "B200": 8.00,
    "GB200": 10.00,
    "GB300": 12.00,
}


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


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


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
    for family in GPU_HOURLY_COST_USD:
        if family in upper:
            return family
    return None


def _build_running_best(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    running_best: list[tuple[int, float]] = []
    best = None
    for step, reward in points:
        best = reward if best is None else max(best, reward)
        running_best.append((step, best))
    return running_best


def _build_reward_chart_data(steps: list["DSEStepData"]) -> dict[str, list[Any]] | None:
    if not steps:
        return None

    reward_points = [(step.step, step.reward) for step in steps]
    running_best = _build_running_best(reward_points)
    return {
        "labels": [step.step for step in steps],
        "rewards": [step.reward for step in steps],
        "running_best": [reward for _, reward in running_best],
        "observations": [step.observation_display for step in steps],
    }


def _build_effort_chart_data(executed_steps: int, total_space: int) -> dict[str, Any] | None:
    if total_space <= 0:
        return None

    explored_ratio = min(max(executed_steps / total_space, 0.0), 1.0)
    explored_display_percent = 100.0 if explored_ratio >= 1.0 else min(max(explored_ratio * 100.0, 14.0), 62.0)

    return {
        "explored_ratio": explored_ratio,
        "explored_display_percent": explored_display_percent,
        "remainder_display_percent": max(100.0 - explored_display_percent, 0.0),
    }


class _ReportMetadataSystem(BaseModel):
    gpu_arch_type: str


class _ReportMetadataSlurm(BaseModel):
    node_list: str


class _ReportSystemMetadata(BaseModel):
    system: _ReportMetadataSystem
    slurm: _ReportMetadataSlurm


class _ReportJobMetadata(BaseModel):
    elapsed_time_sec: int


def load_system_metadata(run_dir: Path, results_root: Path) -> _ReportSystemMetadata | None:
    """Load system metadata from run_dir. At the moment it supports only Slurm."""
    metadata_path = run_dir / "metadata"
    if not metadata_path.exists():
        logging.debug(f"No metadata folder found in {run_dir=}")
        if not (results_root / "metadata").exists():
            logging.debug(f"No metadata folder found in {results_root=}")
            return None
        metadata_path = results_root / "metadata"

    node_files = list(metadata_path.glob("node-*.toml"))
    if not node_files:
        logging.debug(f"No node files found in {metadata_path}")
        return None

    node_file = node_files[0]
    with node_file.open() as f:
        try:
            return _ReportSystemMetadata.model_validate(toml.load(f))
        except Exception as e:
            logging.debug(f"Error validating metadata for {node_file}: {e}")

    return None


@dataclass
class ReportItem:
    """Basic report item for general systems."""

    name: str
    description: str
    logs_path: str | None
    nodes: _ReportSystemMetadata | None
    status_text: str
    status_class: str

    @classmethod
    def from_test_runs(cls, test_runs: list[TestRun], results_root: Path) -> list["ReportItem"]:
        report_items: list[ReportItem] = []
        for tr in test_runs:
            tr_status = tr.test.was_run_successful(tr)
            status_text = "PASSED" if tr_status.is_successful else "FAILED"
            report_items.append(
                ReportItem(
                    name=case_name(tr),
                    description=tr.test.description,
                    logs_path=f"./{tr.output_path.relative_to(results_root)}" if tr.output_path.exists() else None,
                    nodes=load_system_metadata(tr.output_path, results_root),
                    status_text=status_text,
                    status_class=status_text.lower(),
                )
            )
        return report_items


@dataclass
class DSEStepData:
    """DSE step data."""

    step: int
    reward: float
    observation: list[Any]
    observation_display: str
    action: dict[str, Any]
    elapsed_time_sec: int | None = None
    is_successful: bool = False


@dataclass
class DSEParameterRow:
    """DSE parameter row."""

    name: str
    values: list[str]
    best_value: str


@dataclass
class DSESummary:
    """Summary of a DSE iteration."""

    name: str
    description: str
    iteration: int
    output_root: Path
    output_root_rel_path: str
    total_space: int
    executed_steps: int
    skipped_steps: int
    best_step: int | None
    best_reward: float | None
    avg_step_duration_sec: float | None
    total_runtime_sec: float | None
    saved_runtime_sec: float | None
    success_count: int
    failure_count: int
    gpu_arch_label: str | None
    saved_gpu_hours: float | None
    estimated_saved_cost_usd: float | None
    best_config_rel_path: str | None
    best_config_toml: str | None
    analysis_rel_path: str | None
    parameter_rows: list[DSEParameterRow] = field(default_factory=list)
    reward_chart_data: dict[str, list[Any]] | None = None
    effort_chart_data: dict[str, Any] | None = None

    @property
    def display_name(self) -> str:
        if self.iteration == 0:
            return self.name
        return f"{self.name} iter={self.iteration}"

    @property
    def status_text(self) -> str:
        if self.failure_count == 0:
            return "PASSED"
        if self.success_count == 0:
            return "FAILED"
        return "PARTIAL"

    @property
    def status_style(self) -> str:
        return {
            "PASSED": "[green]PASSED[/green]",
            "FAILED": "[red]FAILED[/red]",
            "PARTIAL": "[yellow]PARTIAL[/yellow]",
        }[self.status_text]


class DSEReportBuilder:
    """Build DSE summaries and best-config artifacts from generated results."""

    def __init__(self, system: Any, results_root: Path, loaded_test_runs: list[TestRun]):
        self.system = system
        self.results_root = results_root
        self.loaded_test_runs = loaded_test_runs

    @staticmethod
    def best_config_file_name(tr: TestRun) -> str:
        return f"{tr.name}.toml"

    def build(self, original_test_runs: list[TestRun]) -> list[DSESummary]:
        summaries: list[DSESummary] = []
        for tr in original_test_runs:
            if not tr.is_dse_job:
                continue
            summaries.extend(self._build_for_test_run(tr))
        return summaries

    def _build_for_test_run(self, original_tr: TestRun) -> list[DSESummary]:
        summaries: list[DSESummary] = []
        tr_base_dir = self.results_root / original_tr.name
        if not tr_base_dir.exists():
            return summaries

        grouped_trs: dict[int, list[TestRun]] = {}
        for tr in self.loaded_test_runs:
            if tr.name != original_tr.name:
                continue
            grouped_trs.setdefault(tr.current_iteration, []).append(tr)

        iteration_dirs = sorted((d for d in tr_base_dir.iterdir() if d.is_dir()), key=lambda p: int(p.name))
        for iter_dir in iteration_dirs:
            iteration = int(iter_dir.name)
            summary = self._build_iteration_summary(original_tr, iteration, iter_dir, grouped_trs.get(iteration, []))
            if summary is not None:
                summaries.append(summary)
        return summaries

    def _build_iteration_summary(
        self,
        original_tr: TestRun,
        iteration: int,
        iter_dir: Path,
        step_trs: list[TestRun],
    ) -> DSESummary | None:
        trajectory_file = iter_dir / "trajectory.csv"
        if not trajectory_file.exists():
            logging.warning(f"No trajectory file found for {original_tr.name} at {trajectory_file}")
            return None

        df = lazy.pd.read_csv(trajectory_file)
        if df.empty:
            return None

        steps_by_number = {tr.step: tr for tr in step_trs}
        steps: list[DSEStepData] = []
        for row in df.to_dict(orient="records"):
            step_no = int(row["step"])
            action = _safe_literal_eval(row.get("action"), {})
            if not isinstance(action, dict):
                action = {}
            observation = _safe_literal_eval(row.get("observation"), [])
            if not isinstance(observation, list):
                observation = [observation]
            tr = steps_by_number.get(step_no)
            is_successful = tr.test.was_run_successful(tr).is_successful if tr is not None else False
            steps.append(
                DSEStepData(
                    step=step_no,
                    reward=float(row["reward"]),
                    observation=observation,
                    observation_display=", ".join(_format_scalar(v) for v in observation) if observation else "n/a",
                    action=action,
                    elapsed_time_sec=self._step_elapsed_time(iter_dir / str(step_no)),
                    is_successful=is_successful,
                )
            )

        if not steps:
            return None

        steps.sort(key=lambda step: step.step)
        best_step_data = max(steps, key=lambda step: step.reward)
        best_step_dir = iter_dir / str(best_step_data.step)
        best_step_details = best_step_dir / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME
        if not best_step_details.exists():
            logging.warning(f"No test run dump found for best DSE step at {best_step_details}")
            return None

        with best_step_details.open() as f:
            trd = TestRunDetails.model_validate(toml.load(f))

        best_config_path = iter_dir / self.best_config_file_name(original_tr)
        with best_config_path.open("w") as f:
            toml.dump(trd.test_definition.model_dump(), f)
        best_config_toml = toml.dumps(trd.test_definition.model_dump())

        elapsed_times = [step.elapsed_time_sec for step in steps if step.elapsed_time_sec is not None]
        avg_step_duration_sec = sum(elapsed_times) / len(elapsed_times) if elapsed_times else None
        total_runtime_sec = sum(elapsed_times) if elapsed_times else None
        total_space = len(original_tr.all_combinations)
        executed_steps = len(steps)
        skipped_steps = max(total_space - executed_steps, 0)
        projected_runtime_sec = avg_step_duration_sec * total_space if avg_step_duration_sec is not None else None
        saved_runtime_sec = (
            max(projected_runtime_sec - total_runtime_sec, 0.0)
            if projected_runtime_sec is not None and total_runtime_sec is not None
            else None
        )

        metadata = load_system_metadata(iter_dir / str(best_step_data.step), self.results_root)
        gpu_arch_label = metadata.system.gpu_arch_type if metadata else None
        gpu_arch_family = _normalize_gpu_family(gpu_arch_label)
        num_nodes = trd.nnodes
        gpus_per_node = getattr(self.system, "gpus_per_node", None)
        total_gpu_hours = (
            (total_runtime_sec / 3600.0) * num_nodes * gpus_per_node
            if total_runtime_sec is not None and gpus_per_node is not None
            else None
        )
        projected_gpu_hours = (
            (projected_runtime_sec / 3600.0) * num_nodes * gpus_per_node
            if projected_runtime_sec is not None and gpus_per_node is not None
            else None
        )
        saved_gpu_hours = (
            max(projected_gpu_hours - total_gpu_hours, 0.0)
            if projected_gpu_hours is not None and total_gpu_hours is not None
            else None
        )
        estimated_saved_cost_usd = (
            saved_gpu_hours * GPU_HOURLY_COST_USD[gpu_arch_family]
            if saved_gpu_hours is not None and gpu_arch_family in GPU_HOURLY_COST_USD
            else None
        )

        success_count = sum(1 for step in steps if step.is_successful)
        failure_count = len(steps) - success_count
        best_action = best_step_data.action
        parameter_rows = [
            DSEParameterRow(
                name=name,
                values=[_format_scalar(value) for value in values],
                best_value=_format_scalar(best_action.get(name, "n/a")),
            )
            for name, values in original_tr.param_space.items()
        ]
        analysis_file = iter_dir / "analysis.csv"

        return DSESummary(
            name=original_tr.name,
            description=original_tr.test.description,
            iteration=iteration,
            output_root=iter_dir,
            output_root_rel_path=f"./{iter_dir.relative_to(self.results_root)}",
            total_space=total_space,
            executed_steps=executed_steps,
            skipped_steps=skipped_steps,
            best_step=best_step_data.step,
            best_reward=best_step_data.reward,
            avg_step_duration_sec=avg_step_duration_sec,
            total_runtime_sec=total_runtime_sec,
            saved_runtime_sec=saved_runtime_sec,
            success_count=success_count,
            failure_count=failure_count,
            gpu_arch_label=gpu_arch_label,
            saved_gpu_hours=saved_gpu_hours,
            estimated_saved_cost_usd=estimated_saved_cost_usd,
            best_config_rel_path=f"./{best_config_path.relative_to(self.results_root)}",
            best_config_toml=best_config_toml,
            analysis_rel_path=f"./{analysis_file.relative_to(self.results_root)}" if analysis_file.exists() else None,
            parameter_rows=parameter_rows,
            reward_chart_data=_build_reward_chart_data(steps),
            effort_chart_data=_build_effort_chart_data(executed_steps, total_space),
        )

    @staticmethod
    def _step_elapsed_time(step_dir: Path) -> int | None:
        slurm_job_path = step_dir / "slurm-job.toml"
        if not slurm_job_path.exists():
            return None

        with slurm_job_path.open() as f:
            metadata = _ReportJobMetadata.model_validate(toml.load(f))
        return metadata.elapsed_time_sec
