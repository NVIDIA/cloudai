# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric extraction and log discovery for Megatron-Bridge workloads."""

import glob
import os
import re
from pathlib import Path
from typing import Optional


def mbridge_log_candidates(output_path: os.PathLike[str]) -> list[Path]:
    """Resolve MBridge logs from NeMo-Run's job registry, with layout-based fallbacks."""
    root = Path(output_path)
    launcher_log = root / "cloudai_megatron_bridge_launcher.log"
    job_id = _mbridge_job_id(root, launcher_log)
    registered_logs = _registered_mbridge_logs(root, job_id)
    if registered_logs:
        return _prefer_allranks_logs(registered_logs)

    slurm_logs = list(root.glob("experiments/**/log*.out"))
    slurm_logs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if job_id is not None:
        matching_logs = [path for path in slurm_logs if re.search(rf"(?:^|[_-]){job_id}(?:[_-]|\.)", path.name)]
        if matching_logs:
            slurm_logs = matching_logs
    if slurm_logs:
        return _prefer_allranks_logs(slurm_logs)
    return [launcher_log] if launcher_log.is_file() else []


def _mbridge_job_id(root: Path, launcher_log: Path) -> Optional[str]:
    """Read the submitted job ID from CloudAI metadata, falling back to launcher output."""
    slurm_metadata = root / "slurm-job.toml"
    if slurm_metadata.is_file():
        metadata_match = re.search(
            r'^job_id\s*=\s*["\']?(\d+)',
            slurm_metadata.read_text(encoding="utf-8", errors="replace"),
            re.MULTILINE,
        )
        if metadata_match:
            return metadata_match.group(1)
    if launcher_log.is_file():
        launcher_data = launcher_log.read_text(encoding="utf-8", errors="replace")
        job_matches = re.findall(r"(?:Submitted batch job|Job id)\s*:?[ ]*(\d+)", launcher_data, re.IGNORECASE)
        if job_matches:
            return job_matches[-1]
    return None


def _registered_mbridge_logs(root: Path, job_id: Optional[str]) -> list[Path]:
    """Expand the log glob NeMo-Run records for a submitted Slurm job."""
    job_registry = root / ".slurm_jobs"
    if job_id is None or not job_registry.is_file():
        return []
    for line in job_registry.read_text(encoding="utf-8", errors="replace").splitlines():
        registered_id, separator, registered_value = line.partition("=")
        if separator and registered_id.strip() == job_id:
            log_glob = registered_value.split(",", 1)[0].strip()
            return sorted(Path(path) for path in glob.glob(log_glob) if Path(path).is_file())
    return []


def _prefer_allranks_logs(log_paths: list[Path]) -> list[Path]:
    """Match MBridge 0.5+ behavior by preferring aggregated all-ranks logs when available."""
    unique_paths = list(dict.fromkeys(log_paths))
    allranks_paths = [path for path in unique_paths if "allranks" in path.name]
    return allranks_paths or unique_paths


def read_mbridge_metrics(output_path: os.PathLike[str]) -> tuple[Optional[Path], list[float], list[float]]:
    """Read and combine metrics from all logs selected for the completed MBridge job."""
    candidates = mbridge_log_candidates(output_path)
    if not candidates:
        return None, [], []
    log_contents = [path.read_text(encoding="utf-8", errors="replace") for path in candidates]
    step_times_s, gpu_tflops = _extract_mbridge_metrics_from_logs(log_contents)
    return candidates[0], step_times_s, gpu_tflops


def extract_mbridge_metrics(logs: str) -> tuple[list[float], list[float]]:
    """Extract iteration-associated metrics using the union of supported MBridge formats."""
    return _extract_mbridge_metrics_from_logs([logs])


_MBRIDGE_NUMBER = r"([0-9]+(?:\.[0-9]+)?)"
_MBRIDGE_ITERATION_RE = re.compile(r"iteration\s+(\d+)/", re.IGNORECASE)
_MBRIDGE_STEP_TIME_RE = re.compile(rf"Step\s+Time\s*:\s*{_MBRIDGE_NUMBER}\s*s", re.IGNORECASE)
_MBRIDGE_ELAPSED_TIME_RE = re.compile(rf"elapsed time per iteration \(ms\)\s*:\s*{_MBRIDGE_NUMBER}", re.IGNORECASE)
_MBRIDGE_GPU_TFLOPS_RE = re.compile(rf"GPU utilization\s*:\s*{_MBRIDGE_NUMBER}", re.IGNORECASE)


def _extract_mbridge_metrics_from_logs(log_contents: list[str]) -> tuple[list[float], list[float]]:
    """Parse MBridge 0.3+ metric layouts and merge retries by their zero-based iteration."""
    step_times_by_iteration: dict[int, float] = {}
    elapsed_times_by_iteration: dict[int, float] = {}
    gpu_tflops_by_iteration: dict[int, float] = {}

    for content in log_contents:
        local_step_times, local_elapsed_times, local_gpu_tflops = _extract_mbridge_metrics_from_log(content)
        step_times_by_iteration.update(local_step_times)
        elapsed_times_by_iteration.update(local_elapsed_times)
        gpu_tflops_by_iteration.update(local_gpu_tflops)

    # Explicit Step Time is preferred, with the iteration-time marker filling
    # individual gaps. Later retry logs overwrite earlier values for the same step.
    merged_step_times = {**elapsed_times_by_iteration, **step_times_by_iteration}
    step_times_s = [merged_step_times[step] for step in sorted(merged_step_times)][-10:]
    gpu_tflops = [gpu_tflops_by_iteration[step] for step in sorted(gpu_tflops_by_iteration)][-10:]
    return step_times_s, gpu_tflops


def _extract_mbridge_metrics_from_log(content: str) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Extract iteration-keyed step time, elapsed time, and GPU throughput from one log."""
    first_iteration = _MBRIDGE_ITERATION_RE.search(content)
    step_times_precede_iterations = _metric_precedes_first_iteration(_MBRIDGE_STEP_TIME_RE, content, first_iteration)
    gpu_tflops_precede_iterations = _metric_precedes_first_iteration(_MBRIDGE_GPU_TFLOPS_RE, content, first_iteration)
    iterations: list[int] = []
    ordered_step_times: list[float] = []
    ordered_elapsed_times: list[float] = []
    ordered_gpu_tflops: list[float] = []
    step_times: dict[int, float] = {}
    elapsed_times: dict[int, float] = {}
    gpu_tflops: dict[int, float] = {}
    pending_step_time: Optional[float] = None
    pending_gpu_tflops: Optional[float] = None

    for line in content.splitlines():
        step_time_match = _MBRIDGE_STEP_TIME_RE.search(line)
        gpu_tflops_match = _MBRIDGE_GPU_TFLOPS_RE.search(line)
        elapsed_time_match = _MBRIDGE_ELAPSED_TIME_RE.search(line)
        if step_time_match:
            pending_step_time = float(step_time_match.group(1))
            ordered_step_times.append(pending_step_time)
        if gpu_tflops_match:
            pending_gpu_tflops = float(gpu_tflops_match.group(1))
            ordered_gpu_tflops.append(pending_gpu_tflops)
        if elapsed_time_match:
            ordered_elapsed_times.append(float(elapsed_time_match.group(1)) / 1000.0)

        iteration_match = _MBRIDGE_ITERATION_RE.search(line)
        if iteration_match:
            iteration = int(iteration_match.group(1)) - 1
            iterations.append(iteration)
            if step_times_precede_iterations and pending_step_time is not None:
                step_times[iteration] = pending_step_time
                pending_step_time = None
            if gpu_tflops_precede_iterations and pending_gpu_tflops is not None:
                gpu_tflops[iteration] = pending_gpu_tflops
                pending_gpu_tflops = None
            if elapsed_time_match:
                elapsed_times[iteration] = float(elapsed_time_match.group(1)) / 1000.0

    _fill_missing_mbridge_metrics(step_times, iterations, ordered_step_times)
    _fill_missing_mbridge_metrics(elapsed_times, iterations, ordered_elapsed_times)
    _fill_missing_mbridge_metrics(gpu_tflops, iterations, ordered_gpu_tflops)
    return step_times, elapsed_times, gpu_tflops


def _metric_precedes_first_iteration(
    metric_pattern: re.Pattern[str], content: str, first_iteration: Optional[re.Match[str]]
) -> bool:
    """Detect the 0.3 layout where a metric is pending until the following iteration marker."""
    first_metric = metric_pattern.search(content)
    return first_metric is not None and first_iteration is not None and first_metric.start() <= first_iteration.start()


def _fill_missing_mbridge_metrics(target: dict[int, float], iterations: list[int], values: list[float]) -> None:
    """Apply MBridge 0.4+'s ordered zip behavior without replacing proximity matches."""
    for iteration, value in zip(iterations, values, strict=False):
        target.setdefault(iteration, value)
