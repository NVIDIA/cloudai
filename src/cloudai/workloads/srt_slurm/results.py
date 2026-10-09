# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def sa_bench_metrics(data: dict[str, Any], tdef: "SRTSlurmTestDefinition") -> dict[str, float]:
    metrics = numeric_fields(data)
    aliases = {"max_concurrency": "concurrency", "output_throughput": "throughput_toks"}
    for old, new in aliases.items():
        if old in metrics:
            metrics[new] = metrics[old]
    if "throughput_toks" not in metrics:
        return {}
    metrics["output_throughput"] = metrics["throughput_toks"]
    if tdef.serving_gpus and "total_token_throughput" in metrics:
        metrics["tput_per_gpu"] = metrics["total_token_throughput"] / tdef.serving_gpus
        output_gpus = tdef.decode_gpus or tdef.serving_gpus
        metrics["output_tput_per_gpu"] = metrics["throughput_toks"] / output_gpus
        input_gpus = tdef.prefill_gpus or tdef.serving_gpus
        metrics["input_tput_per_gpu"] = (metrics["total_token_throughput"] - metrics["throughput_toks"]) / input_gpus
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


def agentx_metrics(data: dict[str, Any]) -> dict[str, float]:
    request = data.get("request_metrics")
    if not isinstance(request, dict):
        return {}
    throughput = request.get("throughput", {})
    per_gpu = throughput.get("per_gpu", {}) if isinstance(throughput, dict) else {}
    if not isinstance(per_gpu, dict) or number(per_gpu.get("total_tput_tps")) is None:
        return {}
    metrics = numeric_fields(data)
    if "conc" in metrics:
        metrics["concurrency"] = metrics["conc"]
    for source, dest in (
        ("total_tput_tps", "tput_per_gpu"),
        ("output_tput_tps", "output_tput_per_gpu"),
        ("input_tput_tps", "input_tput_per_gpu"),
    ):
        if (value := number(per_gpu.get(source))) is not None:
            metrics[dest] = value
    metrics.update(numeric_fields(throughput))
    for source, dest in (("total", "total_tput_tps"), ("input", "input_tput_tps"), ("output", "output_tput_tps")):
        block = throughput.get(source, {})
        if isinstance(block, dict) and (value := number(block.get("tokens_per_second"))) is not None:
            metrics[dest] = value
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


def load_results(tdef: "SRTSlurmTestDefinition", tr: TestRun) -> list[dict[str, float]]:
    """Read only the current job's outputs, retaining each concurrency."""
    logs = tdef.logs_path(tr)
    if logs is None:
        return []
    if tdef.cmd_args.result_format == "agentx":
        results = []
        for path in sorted((logs / "agentic").glob("*.json")):
            metrics = agentx_metrics(read_json(path))
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
        return [metrics for path in raw_files if (metrics := sa_bench_metrics(read_json(path), tdef))]
    runs = read_json(logs / "benchmark-rollup.json").get("runs", [])
    if not isinstance(runs, list):
        return []
    return [metrics for run in runs if isinstance(run, dict) and (metrics := sa_bench_metrics(run, tdef))]
