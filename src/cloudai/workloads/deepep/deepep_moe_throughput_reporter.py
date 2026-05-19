# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scenario-level MoE throughput summary: standalone SVG file."""

from __future__ import annotations

import html
import json
import logging
import re
from pathlib import Path

from cloudai._core.base_reporter import Reporter
from cloudai.workloads.deepep.deepep import DeepEPTestDefinition
from cloudai.workloads.deepep.deepep_combined_report import deepep_results_json_files
from cloudai.workloads.nccl_test.nccl import NCCLTestDefinition
from cloudai.workloads.nccl_test.performance_report_generation_strategy import extract_nccl_data
from cloudai.workloads.ucc_test.ucc import UCCTestDefinition


def _deepep_dispatch_combine_bars(test_output: Path) -> list[tuple[str, float, str]]:
    """From latest ``results.json``: one bar per ``dispatch`` / ``combine`` row (``bus_bw_avg``)."""
    paths = deepep_results_json_files(test_output)
    if not paths:
        return []
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    try:
        rows = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logging.debug("DeepEP results.json unreadable %s: %s", latest, e)
        return []
    if not isinstance(rows, list):
        return []

    by_op: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        op = row.get("operation")
        if not isinstance(op, str) or "bus_bw_avg" not in row:
            continue
        op_l = op.lower()
        if op_l not in ("dispatch", "combine"):
            continue
        try:
            by_op[op_l] = float(row["bus_bw_avg"])
        except (TypeError, ValueError):
            continue

    out: list[tuple[str, float, str]] = []
    # Stable order: dispatch then combine, only if present in JSON
    if "dispatch" in by_op:
        out.append(("DeepEP dispatch", by_op["dispatch"], "#2ca02c"))
    if "combine" in by_op:
        out.append(("DeepEP combine", by_op["combine"], "#31a354"))
    return out


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
    """Mean of ``Bus Bandwidth … avg`` column over numeric data rows (8 fields)."""
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
        if sz < 1048576:
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
            vals.append(float(parts[7]))
        except (IndexError, ValueError):
            continue
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _write_moe_throughput_svg(
    path: Path,
    *,
    scenario_name: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    y_axis_label: str,
) -> None:
    """Bar chart + value markers; standalone SVG."""
    n = len(labels)
    ml, mr, mt = 72, 44, 72
    ih = 300
    mb = max(100, 36 + n * 18)
    h = mt + ih + mb
    w = max(720, min(1280, ml + mr + max(1, n) * 92))

    iw = w - ml - mr
    y0 = mt + ih
    vmin, vmax = 0.0, max(values) * 1.12 if values else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    def ypx(v: float) -> float:
        return y0 - (v - vmin) / (vmax - vmin) * ih

    slot = iw / max(n, 1)
    bar_w = min(56.0, slot * 0.55)
    centers = [ml + (i + 0.5) * slot for i in range(n)]
    pts = [(cx, ypx(v)) for cx, v in zip(centers, values, strict=True)]

    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="{ml}" y="36" font-size="16" font-weight="600" fill="#111">{html.escape(scenario_name)}</text>',
        f'<text x="{ml}" y="54" font-size="11" fill="#555">{html.escape(y_axis_label)}</text>',
        f'<line x1="{ml}" y1="{y0}" x2="{ml + iw}" y2="{y0}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{y0}" stroke="#333" stroke-width="1.5"/>',
    ]

    for g in (0.25, 0.5, 0.75):
        gy = y0 - g * ih
        parts.append(
            f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml + iw}" y2="{gy:.1f}" stroke="#ddd" stroke-width="1"/>'
        )
        gv = vmin + g * (vmax - vmin)
        parts.append(
            f'<text x="{ml - 8}" y="{gy + 4:.1f}" font-size="11" text-anchor="end" fill="#444">{gv:.1f}</text>'
        )

    for cx, val, col, lab in zip(centers, values, colors, labels, strict=True):
        top = ypx(val)
        x1 = cx - bar_w / 2
        hbar = y0 - top
        parts.append(
            f'<rect x="{x1:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{hbar:.1f}" '
            f'fill="{html.escape(col)}" fill-opacity="0.35" stroke="{html.escape(col)}" stroke-width="1.5"/>'
        )

    for (cx, cy), val, col, lab in zip(pts, values, colors, labels, strict=True):
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{html.escape(col)}" '
            f'stroke="#222" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{cy - 14:.1f}" font-size="12" text-anchor="middle" '
            f'font-weight="600" fill="#111">{val:.2f}</text>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{y0 + 22:.1f}" font-size="13" text-anchor="middle" fill="#111">'
            f"{html.escape(lab)}</text>"
        )

    parts.append(
        f'<text transform="translate(20 {mt + ih / 2:.1f}) rotate(-90)" '
        f'font-size="12" text-anchor="middle" fill="#444">{html.escape(y_axis_label)}</text>'
    )

    leg_y = y0 + 38
    parts.append(f'<text x="{ml}" y="{leg_y}" font-size="11" font-weight="600" fill="#333">Summary</text>')
    for i, (lab, val, col) in enumerate(zip(labels, values, colors, strict=True)):
        parts.append(
            f'<text x="{ml}" y="{leg_y + 16 + i * 16}" font-size="12" fill="#111">'
            f'<tspan font-weight="600" fill="{html.escape(col)}">{html.escape(lab)}</tspan>'
            f"<tspan>: {val:.4f} GB/s</tspan></text>"
        )

    parts.append("</svg>")

    path.write_text("\n".join(parts), encoding="utf-8")


class DeepEPMoEThroughputReporter(Reporter):
    """After the scenario finishes, write one standalone SVG chart under the results root."""

    def generate(self) -> None:
        self.load_test_runs()
        deepep_trs = [tr for tr in self.trs if isinstance(tr.test, DeepEPTestDefinition)]
        if not deepep_trs:
            logging.debug("Skipping deepep_moe_throughput: no DeepEP test in scenario.")
            return

        categories: list[str] = []
        values: list[float] = []
        colors: list[str] = []

        deepep_bars = _deepep_dispatch_combine_bars(deepep_trs[0].output_path)
        if not deepep_bars:
            logging.warning(
                "Skipping deepep_moe_throughput: no dispatch/combine bus_bw_avg in DeepEP results.json under %s",
                deepep_trs[0].output_path,
            )
            return
        for lab, val, col in deepep_bars:
            categories.append(lab)
            values.append(val)
            colors.append(col)

        ucc_trs = [tr for tr in self.trs if isinstance(tr.test, UCCTestDefinition)]
        if ucc_trs:
            uval = _mean_ucc_bus_bw_gb_s(ucc_trs[0].output_path)
            if uval is not None:
                categories.append("UCC")
                values.append(uval)
                colors.append("#1f77b4")
            else:
                logging.debug("UCC test present but bus bandwidth not parsed from outputs.")

        nccl_trs = [tr for tr in self.trs if isinstance(tr.test, NCCLTestDefinition)]
        if nccl_trs:
            nval = _mean_nccl_oop_busbw_gb_s(nccl_trs[0].output_path)
            if nval is not None:
                categories.append("NCCL")
                values.append(nval)
                colors.append("#ff7f0e")
            else:
                logging.debug("NCCL test present but perf table not parsed from stdout.")

        out = self.results_root / f"{self.test_scenario.name}-moe-throughput.svg"
        _write_moe_throughput_svg(
            out,
            scenario_name=self.test_scenario.name,
            labels=categories,
            values=values,
            colors=colors,
            y_axis_label="Mean bus bandwidth (GB/s)",
        )
        logging.info("Generated MoE throughput comparison at %s", out)
