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

"""MoE benchmark dependency helpers for Slurm UCC/NCCL."""

from __future__ import annotations

from pathlib import Path

from cloudai.core import TestRun

MOE_BENCHMARK_PREV_MOUNT = "/cloudai_moe_benchmark_prev"


def start_post_comp_chain(test_run: TestRun) -> list[TestRun]:
    """Follow ``start_post_comp`` (e.g. UCC -> NCCL -> MoE benchmark)."""
    dep = test_run.dependencies.get("start_post_comp")
    if dep is None:
        return []
    chain: list[TestRun] = []
    seen: set[int] = set()
    cur: TestRun | None = dep.test_run
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(cur)
        nxt = cur.dependencies.get("start_post_comp")
        cur = nxt.test_run if nxt else None
    return chain


def _has_ucc_matrix_under(root: Path) -> bool:
    if (root / "ucc_matrix.txt").is_file():
        return True
    return any(root.glob("**/ucc_matrix.txt"))


def moe_benchmark_root(test_run: TestRun) -> Path | None:
    """MoE benchmark job directory (``ucc_matrix`` or BENCHMARK stdout), walking ``start_post_comp``."""
    for tr in start_post_comp_chain(test_run):
        root = tr.output_path
        if _has_ucc_matrix_under(root):
            return root
        st = root / "stdout.txt"
        if st.is_file():
            try:
                if "BENCHMARK: DeepEP Results" in st.read_text(errors="replace")[:250000]:
                    return root
            except OSError:
                continue
    return None


def moe_benchmark_results_json_files(test_output_path: Path) -> list[Path]:
    """All ``results.json`` paths under ``results/benchmark_*`` or top-level ``benchmark_*``."""
    found: list[Path] = []
    for pattern in ("results/benchmark_*_ranks_*", "benchmark_*_ranks_*"):
        for d in sorted(test_output_path.glob(pattern)):
            if d.is_dir():
                rj = d / "results.json"
                if rj.is_file():
                    found.append(rj)
    return found
