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

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cloudai.core import TestRun

if TYPE_CHECKING:
    from .srt_slurm import SRTSlurmTestDefinition


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def numeric_fields(data: dict[str, Any]) -> dict[str, float]:
    return {key: value for key, raw in data.items() if (value := number(raw)) is not None}


@dataclass(frozen=True)
class JobGPUCounts:
    """Backend GPU pools resolved by srt-slurm for one submitted job."""

    aggregate: int
    prefill: int
    decode: int

    @property
    def serving(self) -> int:
        return self.aggregate + self.prefill + self.decode

    def as_metrics(self) -> dict[str, float]:
        return {
            "serving_gpus": float(self.serving),
            "aggregate_gpus": float(self.aggregate),
            "prefill_gpus": float(self.prefill),
            "decode_gpus": float(self.decode),
        }


def load_gpu_counts(path: Path) -> JobGPUCounts:
    """Read required producer-resolved counts; never infer them from node capacity."""
    if not path.is_file():
        raise ValueError(f"Missing required srt-slurm job metadata: {path}")
    metadata = read_json(path)
    job_id = metadata.get("job_id")
    if not isinstance(job_id, (str, int)) or isinstance(job_id, bool) or str(job_id) != path.stem:
        raise ValueError(f"Invalid or mismatched job_id in srt-slurm job metadata: {path}")
    resources = metadata.get("resources")
    if not isinstance(resources, dict):
        raise ValueError(f"Missing resources in srt-slurm job metadata: {path}")
    counts = {}
    for role in ("agg", "prefill", "decode"):
        workers = resources.get(f"{role}_workers")
        if type(workers) is not int or workers < 0:
            raise ValueError(f"Invalid resources.{role}_workers in srt-slurm job metadata: {path}")
        if workers == 0:
            counts[role] = 0
            continue
        per_worker = resources.get(f"gpus_per_{role}")
        if type(per_worker) is not int or per_worker <= 0:
            raise ValueError(f"Invalid resources.gpus_per_{role} in srt-slurm job metadata: {path}")
        counts[role] = workers * per_worker
    result = JobGPUCounts(aggregate=counts["agg"], prefill=counts["prefill"], decode=counts["decode"])
    if result.serving <= 0:
        raise ValueError(f"No serving GPUs in srt-slurm job metadata: {path}")
    return result


def sa_bench_metrics(data: dict[str, Any], gpus: JobGPUCounts) -> dict[str, float]:
    metrics = numeric_fields(data)
    aliases = {"max_concurrency": "concurrency", "output_throughput": "throughput_toks"}
    for old, new in aliases.items():
        if old in metrics:
            metrics[new] = metrics[old]
    if "throughput_toks" not in metrics:
        return {}
    metrics["output_throughput"] = metrics["throughput_toks"]
    # Producer rollups may contain capacity-based normalization; use only raw measurements.
    for key in ("tput_per_gpu", "output_tput_per_gpu", "input_tput_per_gpu"):
        metrics.pop(key, None)
    metrics.update(gpus.as_metrics())
    metrics["output_tput_per_gpu"] = metrics["throughput_toks"] / (gpus.decode or gpus.serving)
    if "total_token_throughput" in metrics:
        metrics["tput_per_gpu"] = metrics["total_token_throughput"] / gpus.serving
        metrics["input_tput_per_gpu"] = (metrics["total_token_throughput"] - metrics["throughput_toks"]) / (
            gpus.prefill or gpus.serving
        )
    add_sa_latencies(data, metrics)
    return metrics


def sa_latency_value(data: dict[str, Any], metrics: dict[str, float], family: str, stat: str) -> float | None:
    value = metrics.get(f"{stat}_{family}_ms", metrics.get(f"{family}_{stat}_ms"))
    if value is not None:
        return value
    percentiles = data.get(f"percentiles_{family}_ms", [])
    if not stat.startswith("p") or not isinstance(percentiles, list):
        return None
    return next(
        (number(v[1]) for v in percentiles if isinstance(v, (list, tuple)) and len(v) == 2 and v[0] == int(stat[1:])),
        None,
    )


def add_sa_latencies(data: dict[str, Any], metrics: dict[str, float]) -> None:
    for family in ("ttft", "tpot", "itl", "e2el"):
        for stat in ("mean", "median", "p75", "p90", "p95", "p99"):
            value = sa_latency_value(data, metrics, family, stat)
            if value is not None:
                metrics[f"{stat}_{family}"] = value / 1000
                metrics[f"{family}_{stat}_ms"] = value
                if family == "tpot" and value > 0:
                    metrics[f"{stat}_intvty"] = 1000 / value


def agentx_metrics(data: dict[str, Any], gpus: JobGPUCounts) -> dict[str, float]:
    request = data.get("request_metrics")
    if not isinstance(request, dict):
        return {}
    throughput = request.get("throughput")
    if not isinstance(throughput, dict):
        return {}
    metrics = numeric_fields(data)
    if "conc" in metrics:
        metrics["concurrency"] = metrics["conc"]
    # AgentX normalizes every stream by total serving GPUs, including disaggregated runs.
    # Its per_gpu values can reflect stale environment metadata after topology DSE.
    for key in ("tput_per_gpu", "output_tput_per_gpu", "input_tput_per_gpu"):
        metrics.pop(key, None)
    metrics.update(gpus.as_metrics())
    for source, raw_key, normalized_key in (
        ("total", "total_tput_tps", "tput_per_gpu"),
        ("input", "input_tput_tps", "input_tput_per_gpu"),
        ("output", "output_tput_tps", "output_tput_per_gpu"),
    ):
        block = throughput.get(source)
        value = number(block.get("tokens_per_second")) if isinstance(block, dict) else None
        if value is None:
            value = number(throughput.get(raw_key))
        if value is not None:
            metrics[raw_key] = value
            metrics[normalized_key] = value / gpus.serving
    if "tput_per_gpu" not in metrics:
        return {}
    if "output_tput_tps" in metrics:
        metrics["output_throughput"] = metrics["throughput_toks"] = metrics["output_tput_tps"]
    add_agentx_latencies(request.get("latency", {}), metrics)
    return metrics


def add_agentx_latencies(latency: Any, metrics: dict[str, float]) -> None:
    if isinstance(latency, dict):
        # InferenceX's published AgentX interactivity uses full-response ITL.
        for family in ("ttft", "tpot", "itl", "e2el"):
            stats = (
                latency.get("full_response_itl", latency.get("itl", {})) if family == "itl" else latency.get(family, {})
            )
            if not isinstance(stats, dict):
                continue
            for stat, value in numeric_fields(stats).items():
                metrics[f"{stat}_{family}"] = value
                metrics[f"{family}_{stat}_ms"] = value * 1000
                if family == "itl" and value > 0 and stat != "std":
                    metrics[f"{stat}_intvty"] = 1 / value


def load_srt_slurm_results(tdef: "SRTSlurmTestDefinition", tr: TestRun) -> list[dict[str, float]]:
    """Read only the current job's outputs, retaining each concurrency."""
    logs = tdef.logs_path(tr)
    if logs is None:
        return []
    metadata = tdef.metadata_path(tr)
    if metadata is None:
        raise ValueError("No submitted srt-slurm job ID for required job metadata")
    gpus = load_gpu_counts(metadata)
    if tdef.cmd_args.result_format == "agentx":
        results = []
        for path in sorted((logs / "agentic").glob("*.json")):
            metrics = agentx_metrics(read_json(path), gpus)
            concurrency = metrics.get("concurrency")
            if concurrency is None or not concurrency.is_integer():
                continue
            profiles = logs / "agentic" / f"conc_{int(concurrency)}" / "aiperf_artifacts"
            if not any(p.stat().st_size > 0 for p in profiles.rglob("profile_export.jsonl")):
                continue
            results.append(metrics)
        return results
    raw_files = sorted(logs.glob("sa-bench_*/results_*.json"))
    if raw_files:
        return [metrics for path in raw_files if (metrics := sa_bench_metrics(read_json(path), gpus))]
    runs = read_json(logs / "benchmark-rollup.json").get("runs", [])
    if not isinstance(runs, list):
        return []
    return [metrics for run in runs if isinstance(run, dict) and (metrics := sa_bench_metrics(run, gpus))]
