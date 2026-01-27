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

import argparse
import csv
import math
import os
from typing import Any, Dict, List


def compute_percentile(sorted_values: List[float], percentile: float) -> float:
    if not sorted_values:
        return float("nan")
    if percentile <= 0:
        return float(sorted_values[0])
    if percentile >= 100:
        return float(sorted_values[-1])
    # Nearest-rank linear interpolation (common in data tools)
    k = (len(sorted_values) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_values[int(k)])
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "avg": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p99": float("nan"),
            "p95": float("nan"),
            "p90": float("nan"),
            "p75": float("nan"),
            "p50": float("nan"),
            "p25": float("nan"),
            "p10": float("nan"),
            "p5": float("nan"),
            "p1": float("nan"),
        }
    sorted_vals = sorted(values)
    avg_val = sum(sorted_vals) / len(sorted_vals)
    return {
        "avg": round(avg_val, 2),
        "min": round(sorted_vals[0], 2),
        "max": round(sorted_vals[-1], 2),
        "p99": round(compute_percentile(sorted_vals, 99), 2),
        "p95": round(compute_percentile(sorted_vals, 95), 2),
        "p90": round(compute_percentile(sorted_vals, 90), 2),
        "p75": round(compute_percentile(sorted_vals, 75), 2),
        "p50": round(compute_percentile(sorted_vals, 50), 2),
        "p25": round(compute_percentile(sorted_vals, 25), 2),
        "p10": round(compute_percentile(sorted_vals, 10), 2),
        "p5": round(compute_percentile(sorted_vals, 5), 2),
        "p1": round(compute_percentile(sorted_vals, 1), 2),
    }


def parse_float_safe(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LMCACHE bench CSV metrics")
    parser.add_argument("input_csv", help="Path to input CSV (e.g., lmcache_bench_output_0.1.csv)")
    parser.add_argument("--output", "-o", help="Path to write summary CSV. Defaults to <input>_summary.csv")
    args = parser.parse_args()

    input_path = args.input_csv
    output_path = args.output or f"{input_path}_summary.csv"

    rows: List[Dict[str, Any]] = []
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Build summaries
    summaries: List[Dict[str, Any]] = []

    def append_summary(metric_name: str, values: List[float]) -> None:
        clean_values = [v for v in values if v is not None and not math.isnan(v)]
        stats = summarize(clean_values)
        summaries.append({"Metric": metric_name, **stats})

    # Summarize all numeric columns present in the CSV
    all_columns: List[str] = list(rows[0].keys()) if rows else []
    for col in all_columns:
        col_values = [parse_float_safe(r.get(col)) for r in rows]
        append_summary(col, col_values)

    fieldnames = [
        "Metric",
        "avg",
        "min",
        "max",
        "p99",
        "p95",
        "p90",
        "p75",
        "p50",
        "p25",
        "p10",
        "p5",
        "p1",
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    print(f"Wrote summary to: {output_path}")


if __name__ == "__main__":
    main()
