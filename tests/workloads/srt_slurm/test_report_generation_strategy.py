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
from pathlib import Path

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.srt_slurm import SRTSlurmReportGenerationStrategy


def test_report_reads_best_run_directly_from_srt_rollup(
    slurm_system: SlurmSystem, tmp_path: Path, make_tdef, write_job_metadata
):
    tdef = make_tdef(tmp_path)
    test_run = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")
    test_run.output_path.mkdir(parents=True)
    tdef.job_id_path(test_run).write_text("12345\n", encoding="utf-8")
    write_job_metadata(tdef, test_run)
    rollup_path = test_run.output_path / "srt_slurm_outputs" / "12345" / "logs" / "benchmark-rollup.json"
    rollup_path.parent.mkdir(parents=True)
    rollup_path.write_text(
        json.dumps(
            {
                "runs": [
                    {"throughput_toks": 10.0, "ttft_mean_ms": 1.0},
                    {"throughput_toks": 42.0, "ttft_mean_ms": 3.5},
                ]
            }
        ),
        encoding="utf-8",
    )
    report = SRTSlurmReportGenerationStrategy(slurm_system, test_run)

    assert report.can_handle_directory()
    assert report.get_metric("default") == 42.0
    assert report.get_metric("ttft_mean_ms") == 3.5
    assert tdef.was_run_successful(test_run).is_successful
