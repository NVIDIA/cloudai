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

import pathlib

import cloudai.core
import cloudai.report_generator.comparison_report
import cloudai.systems.slurm
from cloudai.workloads.nixl_ep import NixlEPCmdArgs, NixlEPComparisonReport, NixlEPTestDefinition


def _nixl_ep_tr(name: str, num_processes_per_node: int) -> cloudai.core.TestRun:
    return cloudai.core.TestRun(
        name=name,
        test=NixlEPTestDefinition(
            name="nixl_ep",
            description="NIXL EP benchmark",
            test_template_name="NixlEP",
            cmd_args=NixlEPCmdArgs(
                docker_image_url="fake://nixl",
                plan="[[0, 1]]",
                num_processes_per_node=num_processes_per_node,
            ),
        ),
        num_nodes=2,
        nodes=[],
    )


def _write_node_log(
    run_dir: pathlib.Path, node_idx: int, combined_bw: float, dispatch_bw: float, combine_bw: float
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"nixl-ep-node-{node_idx}.log").write_text(
        "\n".join(
            [
                f"[rank {node_idx}] Dispatch + combine bandwidth: {combined_bw} GB/s, "
                "avg_t=10 us, min_t=8 us, max_t=12 us",
                f"[rank {node_idx}] Dispatch bandwidth: {dispatch_bw} GB/s | Combine bandwidth: {combine_bw} GB/s",
            ]
        ),
        encoding="utf-8",
    )


def test_nixl_ep_comparison_report_generates_html(slurm_system: cloudai.systems.slurm.SlurmSystem) -> None:
    tr1 = _nixl_ep_tr("NIXL.EP.No-expansion", 1)
    tr2 = _nixl_ep_tr("NIXL.EP.Single-expansion", 2)

    tr1_dir = slurm_system.output_path / tr1.name / "0"
    tr2_dir = slurm_system.output_path / tr2.name / "0"
    _write_node_log(tr1_dir, 0, combined_bw=100.0, dispatch_bw=40.0, combine_bw=45.0)
    _write_node_log(tr1_dir, 1, combined_bw=120.0, dispatch_bw=50.0, combine_bw=55.0)
    _write_node_log(tr2_dir, 0, combined_bw=130.0, dispatch_bw=60.0, combine_bw=65.0)
    _write_node_log(tr2_dir, 1, combined_bw=150.0, dispatch_bw=70.0, combine_bw=75.0)

    report = NixlEPComparisonReport(
        slurm_system,
        cloudai.core.TestScenario(name="nixl-ep-comparison", test_runs=[tr1, tr2]),
        slurm_system.output_path,
        cloudai.report_generator.comparison_report.ComparisonReportConfig(enable=True),
    )

    report.load_test_runs()
    assert len(report.trs) == 2

    tables = report.create_tables(report.group_test_runs())
    dispatch_combine_bandwidth_table = tables[0]
    dispatch_bandwidth_table = tables[1]
    avg_time_table = tables[3]

    assert "case=No-expansion" in str(dispatch_combine_bandwidth_table.columns[1].header)
    assert "case=Single-expansion" in str(dispatch_combine_bandwidth_table.columns[2].header)
    assert list(dispatch_combine_bandwidth_table.columns[0].cells) == ["0", "1"]
    assert list(dispatch_combine_bandwidth_table.columns[1].cells) == ["100.0", "120.0"]
    assert list(dispatch_bandwidth_table.columns[1].cells) == ["40.0", "50.0"]
    assert list(avg_time_table.columns[1].cells) == ["10.0", "10.0"]

    report.generate()

    assert (slurm_system.output_path / "nixl_ep_comparison.html").exists()
