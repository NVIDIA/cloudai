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

import ast
import contextlib
import io
import logging
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import jinja2
import toml
from rich import box
from rich.console import Console
from rich.table import Table

from cloudai.util.lazy_imports import lazy

from .core import CommandGenStrategy, Reporter, TestRun, case_name
from .models.scenario import TestRunDetails
from .systems.slurm import SlurmSystem, SlurmSystemMetadata
from .systems.slurm.slurm_metadata import SlurmJobMetadata

GPU_HOURLY_COST_USD = {
    "H100": 4.50,
    "B200": 8.00,
    "GB200": 10.00,
    "GB300": 12.00,
}


def _safe_literal_eval(raw: Any, default: Any) -> Any:
    if isinstance(raw, str):
        with contextlib.suppress(SyntaxError, ValueError):
            return ast.literal_eval(raw)
    return default


def _format_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _format_duration(seconds: float | None) -> str:
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


def _format_float(value: float | None, precision: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def _format_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


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


def _chart_points(points: list[tuple[int, float]], width: int, height: int, padding: int) -> list[tuple[float, float]]:
    if not points:
        return []

    x_vals = [step for step, _ in points]
    y_vals = [reward for _, reward in points]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)

    x_span = max(max_x - min_x, 1)
    y_span = max(max_y - min_y, 1e-9)
    inner_width = width - 2 * padding
    inner_height = height - 2 * padding

    result = []
    for step, reward in points:
        x = padding + ((step - min_x) / x_span) * inner_width
        y = height - padding - ((reward - min_y) / y_span) * inner_height
        result.append((x, y))
    return result


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _build_reward_chart_svg(steps: list["DSEStepData"]) -> str | None:
    if not steps:
        return None

    width, height, padding = 720, 260, 34
    reward_points = [(step.step, step.reward) for step in steps]
    running_best = _build_running_best(reward_points)
    reward_coords = _chart_points(reward_points, width, height, padding)
    best_coords = _chart_points(running_best, width, height, padding)

    reward_line = _polyline(reward_coords)
    best_line = _polyline(best_coords)
    y_vals = [reward for _, reward in reward_points]
    y_min, y_max = min(y_vals), max(y_vals)

    circles = []
    for step_data, (x, y) in zip(steps, reward_coords, strict=True):
        tooltip = (
            f"Step {step_data.step} | Reward: {_format_float(step_data.reward, 4)}"
            f" | Observation: {step_data.observation_display}"
        )
        circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#1f77b4"><title>{tooltip}</title></circle>')

    return "\n".join(
        [
            f'<svg viewBox="0 0 {width} {height}" class="dse-chart" role="img" '
            f'aria-label="Reward over DSE steps. Min reward {_format_float(y_min, 4)}, '
            f'max reward {_format_float(y_max, 4)}.">',
            f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" '
            'stroke="#94a3b8" stroke-width="1" />',
            f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" '
            'stroke="#94a3b8" stroke-width="1" />',
            f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{reward_line}" />',
            f'<polyline fill="none" stroke="#ef4444" stroke-width="2" stroke-dasharray="6 4" points="{best_line}" />',
            *circles,
            f'<text x="{width / 2:.0f}" y="{height - 6}" text-anchor="middle" fill="#475569" '
            'font-size="12">Step</text>',
            f'<text x="16" y="{height / 2:.0f}" transform="rotate(-90 16 {height / 2:.0f})" '
            'text-anchor="middle" fill="#475569" font-size="12">Reward</text>',
            "</svg>",
        ]
    )


@dataclass
class ReportItem:
    """Basic report item for general systems."""

    name: str
    description: str
    logs_path: Optional[str] = None

    @classmethod
    def from_test_runs(cls, test_runs: list[TestRun], results_root: Path) -> list["ReportItem"]:
        report_items: list[ReportItem] = []
        for tr in test_runs:
            report_items.append(ReportItem(case_name(tr), tr.test.description))
            if tr.output_path.exists():
                report_items[-1].logs_path = f"./{tr.output_path.relative_to(results_root)}"
        return report_items


@dataclass
class SlurmReportItem:
    """Enhanced report item for Slurm systems with node information."""

    name: str
    description: str
    logs_path: Optional[str] = None
    nodes: Optional[str] = None

    @classmethod
    def get_metadata(cls, run_dir: Path, results_root: Path) -> Optional[SlurmSystemMetadata]:
        metadata_path = run_dir / "metadata"
        if not metadata_path.exists():
            logging.debug(f"No metadata folder found in {run_dir=}")
            if not (results_root / "metadata").exists():
                logging.debug(f"No metadata folder found in {results_root=}")
                return None
            else:  # single-sbatch case
                metadata_path = results_root / "metadata"

        node_files = list(metadata_path.glob("node-*.toml"))
        if not node_files:
            logging.debug(f"No node files found in {metadata_path}")
            return None

        node_file = node_files[0]
        with node_file.open() as f:
            try:
                return SlurmSystemMetadata.model_validate(toml.load(f))
            except Exception as e:
                logging.debug(f"Error validating metadata for {node_file}: {e}")

        return None

    @classmethod
    def from_test_runs(cls, test_runs: list[TestRun], results_root: Path) -> list["SlurmReportItem"]:
        report_items: list[SlurmReportItem] = []
        for tr in test_runs:
            ri = SlurmReportItem(case_name(tr), tr.test.description)
            if tr.output_path.exists():
                ri.logs_path = f"./{tr.output_path.relative_to(results_root)}"
            if metadata := cls.get_metadata(tr.output_path, results_root):
                ri.nodes = metadata.slurm.node_list
            report_items.append(ri)

        return report_items


@dataclass
class DSEStepData:
    """DSE step data."""

    step: int
    reward: float
    observation: list[Any]
    observation_display: str
    action: dict[str, Any]
    elapsed_time_sec: float | None = None
    is_successful: bool = False


@dataclass
class DSEParameterRow:
    """DSE parameter row."""

    name: str
    values: list[str]
    best_value: str


@dataclass
class DSESummary:
    """DSE summary report."""

    name: str
    description: str
    iteration: int
    output_root: Path
    output_root_rel_path: str
    total_space: int
    executed_steps: int
    skipped_steps: int
    coverage_percent: float | None
    best_step: int | None
    best_reward: float | None
    best_observation_display: str
    avg_step_duration_sec: float | None
    total_runtime_sec: float | None
    projected_runtime_sec: float | None
    saved_runtime_sec: float | None
    success_count: int
    failure_count: int
    gpu_arch_label: str | None
    gpu_arch_family: str | None
    gpus_per_node: int | None
    num_nodes: int | None
    total_gpu_hours: float | None
    projected_gpu_hours: float | None
    saved_gpu_hours: float | None
    estimated_saved_cost_usd: float | None
    best_config_rel_path: str | None
    best_scenario_rel_path: str | None
    best_scenario_toml: str | None
    analysis_rel_path: str | None
    parameter_rows: list[DSEParameterRow] = field(default_factory=list)
    chart_svg: str | None = None

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


class PerTestReporter(Reporter):
    """Generates reports per test using test-specific reporting strategies."""

    def generate(self) -> None:
        self.load_test_runs()

        for tr in self.trs:
            logging.debug(f"Available reports: {[r.__name__ for r in tr.reports]} for directory: {tr.output_path}")
            for reporter in tr.reports:
                rgs = reporter(self.system, tr)

                if not rgs.can_handle_directory():
                    logging.warning(f"Skipping '{tr.output_path}', can't handle with strategy={reporter.__name__}.")
                    continue
                try:
                    rgs.generate_report()
                except Exception as e:
                    logging.warning(
                        f"Error generating report for '{tr.output_path}' with strategy={reporter.__name__}: {e}"
                    )


class StatusReporter(Reporter):
    """Generates HTML status reports with system-specific templates."""

    def __init__(self, system, test_scenario, results_root, config):
        super().__init__(system, test_scenario, results_root, config)
        self.dse_summaries: list[DSESummary] = []

    @property
    def template_file_path(self) -> Path:
        return Path(__file__).parent / "util"

    @property
    def template_file(self) -> str:
        if isinstance(self.system, SlurmSystem):
            return "general-slurm-report.jinja2"
        return "general-report.jinja2"

    def best_dse_config_file_name(self, tr: TestRun) -> str:
        return f"{tr.name}.toml"

    def best_dse_scenario_file_name(self, tr: TestRun) -> str:
        return f"{tr.name}-best-in-scenario.toml"

    def generate(self) -> None:
        self.load_test_runs()
        self.report_best_dse_config()
        self.generate_scenario_report()
        self.print_summary()

    def generate_scenario_report(self) -> None:
        template = jinja2.Environment(loader=jinja2.FileSystemLoader(self.template_file_path)).get_template(
            self.template_file
        )

        report_items = (
            SlurmReportItem.from_test_runs(self.trs, self.results_root)
            if isinstance(self.system, SlurmSystem)
            else ReportItem.from_test_runs(self.trs, self.results_root)
        )
        report = template.render(
            name=self.test_scenario.name,
            report_items=report_items,
            dse_summaries=self.dse_summaries,
            format_duration=_format_duration,
            format_float=_format_float,
            format_percent=_format_percent,
            format_money=_format_money,
        )
        report_path = self.results_root / f"{self.test_scenario.name}.html"
        with report_path.open("w") as f:
            f.write(report)

        logging.info(f"Generated scenario report at {report_path}")

    def report_best_dse_config(self):
        self.dse_summaries = []
        for tr in self.test_scenario.test_runs:
            if not tr.is_dse_job:
                continue

            self.dse_summaries.extend(self._build_dse_summaries(tr))

    def _build_dse_summaries(self, original_tr: TestRun) -> list[DSESummary]:
        summaries: list[DSESummary] = []
        tr_base_dir = self.results_root / original_tr.name
        if not tr_base_dir.exists():
            return summaries

        grouped_trs: dict[int, list[TestRun]] = {}
        for tr in self.trs:
            if tr.name != original_tr.name:
                continue
            grouped_trs.setdefault(tr.current_iteration, []).append(tr)
        iteration_dirs = sorted((d for d in tr_base_dir.iterdir() if d.is_dir()), key=lambda p: int(p.name))
        for iter_dir in iteration_dirs:
            iteration = int(iter_dir.name)
            summary = self._build_dse_summary_for_iteration(
                original_tr, iteration, iter_dir, grouped_trs.get(iteration, [])
            )
            if summary is not None:
                summaries.append(summary)
        return summaries

    def _build_dse_summary_for_iteration(
        self, original_tr: TestRun, iteration: int, iter_dir: Path, step_trs: list[TestRun]
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
            elapsed_time = self._step_elapsed_time(iter_dir / str(step_no))
            tr = steps_by_number.get(step_no)
            is_successful = tr.test.was_run_successful(tr).is_successful if tr is not None else False
            steps.append(
                DSEStepData(
                    step=step_no,
                    reward=float(row["reward"]),
                    observation=observation,
                    observation_display=", ".join(_format_scalar(v) for v in observation) if observation else "n/a",
                    action=action,
                    elapsed_time_sec=elapsed_time,
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

        best_config_path = iter_dir / self.best_dse_config_file_name(original_tr)
        logging.info(f"Writing best config for {original_tr.name} to {best_config_path}")
        with best_config_path.open("w") as f:
            toml.dump(trd.test_definition.model_dump(), f)

        best_scenario_content = self._build_best_scenario_toml(original_tr, trd)
        best_scenario_path = iter_dir / self.best_dse_scenario_file_name(original_tr)
        with best_scenario_path.open("w") as f:
            f.write(best_scenario_content)

        elapsed_times = [step.elapsed_time_sec for step in steps if step.elapsed_time_sec is not None]
        avg_step_duration_sec = sum(elapsed_times) / len(elapsed_times) if elapsed_times else None
        total_runtime_sec = sum(elapsed_times) if elapsed_times else None
        total_space = len(original_tr.all_combinations)
        executed_steps = len(steps)
        skipped_steps = max(total_space - executed_steps, 0)
        coverage_percent = (executed_steps / total_space * 100.0) if total_space else None
        projected_runtime_sec = avg_step_duration_sec * total_space if avg_step_duration_sec is not None else None
        saved_runtime_sec = (
            max(projected_runtime_sec - total_runtime_sec, 0.0)
            if projected_runtime_sec is not None and total_runtime_sec is not None
            else None
        )

        metadata = self._best_available_metadata(iter_dir, best_step_data.step)
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
            coverage_percent=coverage_percent,
            best_step=best_step_data.step,
            best_reward=best_step_data.reward,
            best_observation_display=best_step_data.observation_display,
            avg_step_duration_sec=avg_step_duration_sec,
            total_runtime_sec=total_runtime_sec,
            projected_runtime_sec=projected_runtime_sec,
            saved_runtime_sec=saved_runtime_sec,
            success_count=success_count,
            failure_count=failure_count,
            gpu_arch_label=gpu_arch_label,
            gpu_arch_family=gpu_arch_family,
            gpus_per_node=gpus_per_node,
            num_nodes=num_nodes,
            total_gpu_hours=total_gpu_hours,
            projected_gpu_hours=projected_gpu_hours,
            saved_gpu_hours=saved_gpu_hours,
            estimated_saved_cost_usd=estimated_saved_cost_usd,
            best_config_rel_path=f"./{best_config_path.relative_to(self.results_root)}",
            best_scenario_rel_path=f"./{best_scenario_path.relative_to(self.results_root)}",
            best_scenario_toml=best_scenario_content,
            analysis_rel_path=f"./{analysis_file.relative_to(self.results_root)}" if analysis_file.exists() else None,
            parameter_rows=parameter_rows,
            chart_svg=_build_reward_chart_svg(steps),
        )

    def _build_best_scenario_toml(self, original_tr: TestRun, best_trd: TestRunDetails) -> str:
        tdef = best_trd.test_definition.model_copy(deep=True)
        tdef.agent = None
        tdef.agent_steps = None
        tdef.agent_reward_function = None
        tdef.agent_config = None
        tdef.agent_metrics = ["default"]

        test_block: dict[str, Any] = {
            "id": original_tr.name,
            "num_nodes": best_trd.nnodes,
            "name": tdef.name,
            "description": tdef.description,
            "test_template_name": tdef.test_template_name,
            "cmd_args": tdef.cmd_args.model_dump(by_alias=True),
        }
        if original_tr.time_limit:
            test_block["time_limit"] = original_tr.time_limit
        if original_tr.nodes:
            test_block["nodes"] = original_tr.nodes
        if original_tr.exclude_nodes:
            test_block["exclude_nodes"] = original_tr.exclude_nodes
        if tdef.extra_env_vars:
            test_block["extra_env_vars"] = tdef.extra_env_vars
        if tdef.extra_container_mounts:
            test_block["extra_container_mounts"] = tdef.extra_container_mounts
        if tdef.git_repos:
            test_block["git_repos"] = [repo.model_dump() for repo in tdef.git_repos]
        if tdef.nsys:
            test_block["nsys"] = tdef.nsys.model_dump(exclude_unset=True)
        if original_tr.extra_srun_args:
            test_block["extra_srun_args"] = original_tr.extra_srun_args

        scenario_dict = {
            "name": f"{best_trd.test_definition.name}_best_config",
            "Tests": [test_block],
        }
        buffer = io.StringIO()
        toml.dump(scenario_dict, buffer)
        return buffer.getvalue()

    @staticmethod
    def _step_elapsed_time(step_dir: Path) -> float | None:
        slurm_job_path = step_dir / "slurm-job.toml"
        if not slurm_job_path.exists():
            return None

        with slurm_job_path.open() as f:
            metadata = SlurmJobMetadata.model_validate(toml.load(f))
        return float(metadata.elapsed_time_sec)

    def _best_available_metadata(self, iter_dir: Path, best_step: int) -> SlurmSystemMetadata | None:
        if not isinstance(self.system, SlurmSystem):
            return None
        best_step_dir = iter_dir / str(best_step)
        return SlurmReportItem.get_metadata(best_step_dir, self.results_root)

    def print_summary(self) -> None:
        if not self.trs:
            logging.debug("No test runs found, skipping summary.")
            return

        table = Table(title="Scenario results", title_justify="left", show_lines=True, box=box.DOUBLE_EDGE)
        for col in ["Case", "Status", "Details"]:
            table.add_column(col, overflow="fold")

        if self.dse_summaries:
            for summary in self.dse_summaries:
                details = [
                    f"steps={summary.executed_steps}/{summary.total_space}",
                    f"best_step={summary.best_step}",
                    f"best_reward={_format_float(summary.best_reward, 4)}",
                    f"failures={summary.failure_count}",
                ]
                if summary.best_scenario_rel_path:
                    details.append(summary.best_scenario_rel_path)
                table.add_row(summary.display_name, f"[bold]{summary.status_style}[/bold]", "\n".join(details))
        else:
            for tr in self.trs:
                tr_status = tr.test.was_run_successful(tr)
                sts_text = f"[bold]{'[green]PASSED[/green]' if tr_status.is_successful else '[red]FAILED[/red]'}[/bold]"
                display_path = str(tr.output_path.absolute())
                with contextlib.suppress(ValueError):
                    display_path = str(tr.output_path.absolute().relative_to(Path.cwd()))
                details_text = f"\n{tr_status.error_message}" if tr_status.error_message else ""
                columns = [tr.name, sts_text, f"{display_path}{details_text}"]
                table.add_row(*columns)

        console = Console()
        with console.capture() as capture:
            console.print(table)  # doesn't print to stdout, captures only

        logging.info(capture.get())


class TarballReporter(Reporter):
    """Creates tarballs of results for failed test runs."""

    def generate(self) -> None:
        self.load_test_runs()

        if any(not self.is_successful(tr) for tr in self.trs):
            self.create_tarball(self.results_root)

    def is_successful(self, tr: TestRun) -> bool:
        return tr.test.was_run_successful(tr).is_successful

    def create_tarball(self, directory: Path) -> None:
        tarball_path = Path(str(directory) + ".tgz")
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(directory, arcname=directory.name)
        logging.info(f"Created tarball at {tarball_path}")
