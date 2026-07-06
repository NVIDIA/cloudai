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

"""
Scenario-level MoE throughput summary: one standalone SVG with a 2x2 panel grid.

TL  Bus BW — dispatch       TR  Bus BW — combine        (all backends + UCC/NCCL)
BL  NVLink vs RDMA — disp.  BR  NVLink vs RDMA — comb.   (EP backends only)
"""

from __future__ import annotations

import html
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from cloudai.core import Reporter
from cloudai.workloads.common.moe_benchmark_report import moe_benchmark_results_json_files
from cloudai.workloads.moe_benchmark.moe_benchmark import MoEBenchmarkTestDefinition
from cloudai.workloads.nccl_test.nccl import NCCLTestDefinition
from cloudai.workloads.nccl_test.performance_report_generation_strategy import extract_nccl_data
from cloudai.workloads.ucc_test.ucc import UCCTestDefinition


def _read_moe_results_rows(test_output: Path) -> list[object]:
    """
    Read and concatenate rows from ALL results.json under the test output.

    One row per backend x operation; backends run as separate processes that append/merge.
    """
    rows: list[object] = []
    for path in moe_benchmark_results_json_files(test_output):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logging.debug("MoE benchmark results.json unreadable %s: %s", path, e)
            continue
        if isinstance(data, list):
            rows.extend(data)
    return rows


_DEEPEP_BACKEND_LABEL = {
    "legacy": "legacy",
    "elastic": "elastic",
    "nixl_ep": "nixl_ep",
    "uccl_ep": "uccl_ep",
    "nccl_ep": "nccl_ep",
    "deepep_hybrid": "hybrid_ep",
}
# Per-backend colors (bus panels): legacy green, elastic purple, nixl_ep red,
# uccl_ep cyan, nccl_ep brown, hybrid_ep pink. UCC/NCCL baselines = gray.
_DEEPEP_BACKEND_COLOR = {
    "legacy": "#2ca02c",
    "elastic": "#9467bd",
    "nixl_ep": "#d62728",
    "uccl_ep": "#17becf",
    "nccl_ep": "#8c564b",
    "deepep_hybrid": "#e377c2",
}
_BASELINE_COLOR = "#9a9a9a"  # UCC / NCCL all-to-all-v baselines

# NVLink vs RDMA split colors (bottom panels), one hue per component.
_NVL_COLOR = "#1baf7a"  # NVLink (intra-node, scale-up)
_RDMA_COLOR = "#2a78d6"  # RDMA  (inter-node, scale-out)


def _extract_moe_bus_bw(row: object) -> tuple[str, str, float] | None:
    """Return ``(backend, operation, bus_bw_avg)`` for a dispatch/combine row."""
    if not isinstance(row, dict):
        return None
    op = row.get("operation")
    if not isinstance(op, str) or "bus_bw_avg" not in row:
        return None
    op_l = op.lower()
    if op_l not in ("dispatch", "combine"):
        return None
    backend = row.get("backend")
    backend_l = backend if isinstance(backend, str) else ""
    try:
        val = float(row["bus_bw_avg"])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):  # skip NaN/Inf so gmax stays finite for the int() in the SVG path
        return None
    return backend_l, op_l, val


def _extract_moe_separate_bw(row: object) -> tuple[str, str, float, float] | None:
    """
    Return ``(backend, operation, nvl_bw, rdma_bw)`` for a dispatch/combine row.

    Only EP backends populate ``separate_nvl_bw`` / ``separate_rdma_bw``, so this
    naturally excludes the UCC / NCCL all-to-all-v baselines (single bus bw, no split).
    """
    if not isinstance(row, dict):
        return None
    op = row.get("operation")
    if not isinstance(op, str):
        return None
    op_l = op.lower()
    if op_l not in ("dispatch", "combine"):
        return None
    if "separate_nvl_bw" not in row or "separate_rdma_bw" not in row:
        return None
    backend = row.get("backend")
    backend_l = backend if isinstance(backend, str) else ""
    try:
        nvl = float(row["separate_nvl_bw"])
        rdma = float(row["separate_rdma_bw"])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(nvl) and math.isfinite(rdma)):  # skip NaN/Inf (keeps gmax finite)
        return None
    return backend_l, op_l, nvl, rdma


def _moe_bus_bars(rows: list[object], op_filter: str) -> list[tuple[str, float, str]]:
    """For one operation: ordered ``(backend_label, bus_bw, color)`` per EP backend."""
    order: list[str] = []
    data: dict[str, float] = {}
    for row in rows:
        ex = _extract_moe_bus_bw(row)
        if ex is None:
            continue
        backend, op, val = ex
        if op != op_filter:
            continue
        if backend not in data:
            order.append(backend)
        data[backend] = val
    out: list[tuple[str, float, str]] = []
    for b in order:
        base = b[:-3] if b.endswith("_ll") else b  # low-latency tags are "<backend>_ll"
        out.append((_DEEPEP_BACKEND_LABEL.get(base, base), data[b], _DEEPEP_BACKEND_COLOR.get(base, "#2ca02c")))
    return out


def _moe_nvl_rdma_bars(rows: list[object], op_filter: str) -> list[tuple[str, float, float]]:
    """For one operation: ordered ``(backend_label, nvl_bw, rdma_bw)`` per EP backend."""
    order: list[str] = []
    data: dict[str, tuple[float, float]] = {}
    for row in rows:
        ex = _extract_moe_separate_bw(row)
        if ex is None:
            continue
        backend, op, nvl, rdma = ex
        if op != op_filter:
            continue
        if backend not in data:
            order.append(backend)
        data[backend] = (nvl, rdma)
    return [(_DEEPEP_BACKEND_LABEL.get(b[:-3] if b.endswith("_ll") else b, b), data[b][0], data[b][1]) for b in order]


def _mean_ucc_bus_bw_gb_s(test_output: Path) -> float | None:
    for name in ("stdout.txt", "ucc_perftest_capture.log"):
        path = test_output / name
        if not path.is_file():
            continue
        v = _parse_ucc_perftest_mean_bus_avg(path)
        if v is not None:
            return v
    return None


def _parse_ucc_perftest_mean_bus_avg(path: Path) -> float | None:
    """Mean of ``Bus Bandwidth ... avg`` column over numeric data rows (8 fields)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    avgs: list[float] = []
    for line in text.splitlines():
        parts = re.split(r"\s+", line.strip())
        if len(parts) != 8:
            continue
        if not parts[0].isdigit() or not parts[1].isdigit():
            continue
        try:
            sz = float(parts[1])
            bavg = float(parts[5])
        except ValueError:
            continue
        if sz < 1048576 or not math.isfinite(bavg):
            continue
        avgs.append(bavg)
    if not avgs:
        return None
    return float(sum(avgs) / len(avgs))


def _mean_nccl_oop_busbw_gb_s(test_output: Path) -> float | None:
    rows, _, _, _ = extract_nccl_data(test_output / "stdout.txt")
    if not rows:
        return None
    vals: list[float] = []
    for parts in rows:
        try:
            v = float(parts[7])
        except (IndexError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        vals.append(v)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


# ---------------------------------------------------------------------------
# Panel renderers — each draws one sub-chart inside the rect (x0,y0,pw,ph) of the
# parent SVG, returning a list of SVG fragment strings. Shared internal margins.
# ---------------------------------------------------------------------------
_PML, _PMT, _PMR, _PMB = 48, 46, 16, 54


def _panel_frame(
    x0: float, y0: float, pw: float, ph: float, title: str, subtitle: str, vmax: float
) -> tuple[list[str], float, float, float, float, float]:
    """Title/subtitle + axes + gridlines for a panel; returns (frags, ax0, ay0, ax1, ay1, vmax)."""
    ax0, ay0 = x0 + _PML, y0 + _PMT
    ax1, ay1 = x0 + pw - _PMR, y0 + ph - _PMB
    if vmax <= 0.0:
        vmax = 1.0
    f = [
        f'<text x="{x0 + _PML}" y="{y0 + 18}" font-size="13" font-weight="600" fill="#111">{html.escape(title)}</text>',
        f'<text x="{x0 + _PML}" y="{y0 + 33}" font-size="10" fill="#777">{html.escape(subtitle)}</text>',
        f'<line x1="{ax0}" y1="{ay1}" x2="{ax1}" y2="{ay1}" stroke="#333" stroke-width="1.3"/>',
        f'<line x1="{ax0}" y1="{ay0}" x2="{ax0}" y2="{ay1}" stroke="#333" stroke-width="1.3"/>',
    ]
    ih = ay1 - ay0
    for g in (0.25, 0.5, 0.75, 1.0):
        gy = ay1 - g * ih
        f.append(f'<line x1="{ax0}" y1="{gy:.1f}" x2="{ax1}" y2="{gy:.1f}" stroke="#e2e2e2" stroke-width="1"/>')
        f.append(
            f'<text x="{ax0 - 6}" y="{gy + 4:.1f}" font-size="10" text-anchor="end" fill="#999">{g * vmax:.0f}</text>'
        )
    return f, ax0, ay0, ax1, ay1, vmax


def _xlabel(cx: float, y: float, label: str) -> str:
    return (
        f'<text x="{cx:.1f}" y="{y:.1f}" font-size="10" text-anchor="end" '
        f'transform="rotate(-30 {cx:.1f} {y:.1f})" fill="#333">{html.escape(label)}</text>'
    )


def _panel_single(x0, y0, pw, ph, title, subtitle, entries, vmax) -> list[str]:
    """
    Single-series bar panel: entries = [(label, value, color)].

    `vmax` is shared across all panels so bar heights are directly comparable between charts.
    """
    f, ax0, ay0, ax1, ay1, vmax = _panel_frame(x0, y0, pw, ph, title, subtitle, vmax)
    iw, ih = ax1 - ax0, ay1 - ay0
    n = len(entries)
    slot = iw / max(n, 1)
    bw = min(30.0, slot * 0.6)
    for i, (lab, val, col) in enumerate(entries):
        cx = ax0 + (i + 0.5) * slot
        top = ay1 - (val / vmax) * ih
        f.append(
            f'<rect x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{ay1 - top:.1f}" '
            f'fill="{html.escape(col)}" fill-opacity="0.85"/>'
        )
        f.append(
            f'<text x="{cx:.1f}" y="{top - 4:.1f}" font-size="10" text-anchor="middle" '
            f'font-weight="600" fill="#111">{val:.0f}</text>'
        )
        f.append(_xlabel(cx, ay1 + 12, lab))
    return f


def _panel_grouped(x0, y0, pw, ph, title, subtitle, entries, vmax) -> list[str]:
    """
    Two-series grouped bar panel (NVLink + RDMA): entries = [(label, nvl, rdma)].

    `vmax` is shared across all panels for comparable bar heights.
    """
    f, ax0, ay0, ax1, ay1, vmax = _panel_frame(x0, y0, pw, ph, title, subtitle, vmax)
    iw, ih = ax1 - ax0, ay1 - ay0
    # Legend on the subtitle line (under the panel title), right-aligned.
    ly = y0 + 33
    lx = x0 + pw - _PMR - 188
    f.append(f'<rect x="{lx:.0f}" y="{ly - 9:.0f}" width="11" height="11" fill="{_NVL_COLOR}"/>')
    f.append(f'<text x="{lx + 15:.0f}" y="{ly:.0f}" font-size="10" fill="#111">NVLink</text>')
    f.append(
        f'<rect x="{lx + 78:.0f}" y="{ly - 9:.0f}" width="11" height="11" '
        f'fill="url(#rdmaHatch)" stroke="{_RDMA_COLOR}"/>'
    )
    f.append(f'<text x="{lx + 93:.0f}" y="{ly:.0f}" font-size="10" fill="#111">RDMA</text>')
    n = len(entries)
    slot = iw / max(n, 1)
    bw = min(18.0, slot * 0.34)
    for i, (lab, nvl, rdma) in enumerate(entries):
        gc = ax0 + (i + 0.5) * slot
        for val, fill, stroke, dx in (
            (nvl, _NVL_COLOR, _NVL_COLOR, -0.56),
            (rdma, "url(#rdmaHatch)", _RDMA_COLOR, 0.56),
        ):
            bx = gc + dx * bw - bw / 2
            top = ay1 - (val / vmax) * ih
            f.append(
                f'<rect x="{bx:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{ay1 - top:.1f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
            )
            f.append(
                f'<text x="{bx + bw / 2:.1f}" y="{top - 4:.1f}" font-size="9.5" text-anchor="middle" '
                f'font-weight="600" fill="#111">{val:.0f}</text>'
            )
        f.append(_xlabel(gc, ay1 + 12, lab))
    return f


def _write_dashboard_svg(
    path: Path,
    *,
    scenario_name: str,
    bus_dispatch: list[tuple[str, float, str]],
    bus_combine: list[tuple[str, float, str]],
    nvl_dispatch: list[tuple[str, float, float]],
    nvl_combine: list[tuple[str, float, float]],
    subtitle: str = "",
) -> None:
    """
    One SVG dashboard, laid out as a grid of ONLY the panels that have data.

    Columns = operations present (dispatch, and combine if measured); rows = metrics
    present ("Bus BW" always; "NVLink vs RDMA" only when the backend reports the split —
    e.g. standard mode does, low-latency mode does not). So HT renders 2x2, a
    dispatch-only HT run renders 2x1, and a low-latency run renders a single bus row.
    All panels share ONE y-axis ceiling (multiple of 50, above the global max) so bars
    are comparable. `subtitle` carries the run config (tokens/hidden/top-k/cluster/nodes).
    """
    sub_bus = "all backends · UCC/NCCL = baselines (gray)"
    sub_split = "EP backends only"
    # Rows present: bus always (if any bus data); nvl/rdma only if the split exists.
    rows: list[tuple[str, Any, dict[str, Any], str]] = [
        ("Bus BW", _panel_single, {"dispatch": bus_dispatch, "combine": bus_combine}, sub_bus)
    ]
    if nvl_dispatch or nvl_combine:
        rows.append(("NVLink vs RDMA", _panel_grouped, {"dispatch": nvl_dispatch, "combine": nvl_combine}, sub_split))
    # Columns present: dispatch always; combine only if measured.
    cols = ["dispatch"]
    if bus_combine or nvl_combine:
        cols.append("combine")

    pw, ph, gap, top = 480.0, 340.0, 26.0, 58.0
    ncol, nrow = len(cols), len(rows)
    w = gap * (ncol + 1) + pw * ncol
    h = top + gap * (nrow + 1) + ph * nrow

    allv = (
        [v for _, v, _ in bus_dispatch]
        + [v for _, v, _ in bus_combine]
        + [v for _, n, r in nvl_dispatch for v in (n, r)]
        + [v for _, n, r in nvl_combine for v in (n, r)]
    )
    gmax = max(allv) if allv else 1.0
    vmax = (int(gmax * 1.1 // 50) + 1) * 50.0  # nice ceiling, multiple of 50, ~10% headroom

    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" viewBox="0 0 {w:.0f} {h:.0f}">',
        (
            '<defs><pattern id="rdmaHatch" patternUnits="userSpaceOnUse" width="6" height="6">'
            f'<rect width="6" height="6" fill="{_RDMA_COLOR}"/>'
            '<path d="M0,6 L6,0" stroke="#ffffff" stroke-width="1.2" stroke-opacity="0.55"/>'
            "</pattern></defs>"
        ),
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="{gap}" y="28" font-size="17" font-weight="600" fill="#111">'
        f"{html.escape(scenario_name)} — MoE EP bandwidth</text>",
        f'<text x="{gap}" y="47" font-size="12" fill="#555">{html.escape(subtitle)}</text>',
    ]
    for ri, (row_title, panel_fn, data_by_op, sub) in enumerate(rows):
        for ci, op in enumerate(cols):
            x0 = gap + ci * (pw + gap)
            y0 = top + gap + ri * (ph + gap)
            parts += panel_fn(x0, y0, pw, ph, f"{row_title} — {op}", sub, data_by_op[op], vmax)
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


class MoEBenchmarkThroughputReporter(Reporter):
    """After the scenario finishes, write one standalone 2x2 SVG under the results root."""

    def generate(self) -> None:
        self.load_test_runs()
        moe_trs = [tr for tr in self.trs if isinstance(tr.test, MoEBenchmarkTestDefinition)]
        if not moe_trs:
            logging.debug("Skipping moe_benchmark_throughput: no MoEBenchmark test in scenario.")
            return

        rows = _read_moe_results_rows(moe_trs[0].output_path)
        bus_dispatch = _moe_bus_bars(rows, "dispatch")
        bus_combine = _moe_bus_bars(rows, "combine")
        nvl_dispatch = _moe_nvl_rdma_bars(rows, "dispatch")
        nvl_combine = _moe_nvl_rdma_bars(rows, "combine")

        if not bus_dispatch and not bus_combine:
            logging.warning(
                "Skipping moe_benchmark_throughput: no dispatch/combine bus_bw_avg in results.json under %s",
                moe_trs[0].output_path,
            )
            return

        # Whether the EP backends actually MEASURED combine (benchmark_combine=true). Decide
        # this from EP data ONLY, before injecting baselines — otherwise the baselines below
        # would make bus_combine non-empty and resurrect an (otherwise empty) combine column
        # for a dispatch-only run.
        has_combine = bool(bus_combine or nvl_combine)

        # Append the UCC / NCCL all-to-all-v baselines (single value each, repeated as a
        # reference). They mirror only the operations the EP backends measured: always
        # dispatch, and combine ONLY when there's real EP combine data. No NVLink/RDMA
        # split -> not added to the bottom panels.
        ucc_trs = [tr for tr in self.trs if isinstance(tr.test, UCCTestDefinition)]
        if ucc_trs:
            uval = _mean_ucc_bus_bw_gb_s(ucc_trs[0].output_path)
            if uval is not None:
                bus_dispatch.append(("UCC", uval, _BASELINE_COLOR))
                if has_combine:
                    bus_combine.append(("UCC", uval, _BASELINE_COLOR))
        nccl_trs = [tr for tr in self.trs if isinstance(tr.test, NCCLTestDefinition)]
        if nccl_trs:
            nval = _mean_nccl_oop_busbw_gb_s(nccl_trs[0].output_path)
            if nval is not None:
                bus_dispatch.append(("NCCL", nval, _BASELINE_COLOR))
                if has_combine:
                    bus_combine.append(("NCCL", nval, _BASELINE_COLOR))

        # Run config for the header subtitle: tokens / hidden / top-k / cluster / nodes.
        ca = moe_trs[0].test.cmd_args
        nn = moe_trs[0].num_nodes
        if isinstance(nn, list):
            nn = nn[0] if nn else "?"
        cluster = getattr(self.system, "name", "?")
        subtitle = (
            f"{ca.mode} · tokens={ca.tokens} · hidden={ca.hidden_size} · top-k={ca.num_topk} · "
            f"{ca.data_type} · {cluster} · {nn} nodes"
        )

        out = self.results_root / f"{self.test_scenario.name}-moe-throughput.svg"
        _write_dashboard_svg(
            out,
            scenario_name=self.test_scenario.name,
            subtitle=subtitle,
            bus_dispatch=bus_dispatch,
            bus_combine=bus_combine,
            nvl_dispatch=nvl_dispatch,
            nvl_combine=nvl_combine,
        )
