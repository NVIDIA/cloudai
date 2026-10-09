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

import csv
import json
from pathlib import Path

import pytest

from cloudai.core import METRIC_ERROR, TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.srt_slurm import SRTSlurmReportGenerationStrategy
from cloudai.workloads.srt_slurm.results import agentx_metrics, load_results, sa_bench_metrics

from .test_srt_slurm import make_tdef


def test_fixed_sequence_uses_serving_gpu_counts(tmp_path: Path):
    tdef = make_tdef(tmp_path)
    tdef.serving_gpus = 32
    tdef.prefill_gpus = 24
    tdef.decode_gpus = 8
    metrics = sa_bench_metrics(
        {"total_token_throughput": 300583, "output_throughput": 32000, "median_tpot_ms": 44.59, "median_ttft_ms": 1000},
        tdef,
    )
    assert metrics["tput_per_gpu"] == pytest.approx(9393.21875)
    assert metrics["output_tput_per_gpu"] == 4000
    assert metrics["input_tput_per_gpu"] == pytest.approx((300583 - 32000) / 24)
    assert metrics["median_intvty"] == pytest.approx(1000 / 44.59)
    assert metrics["median_ttft"] == 1


@pytest.mark.parametrize(
    "runs",
    [None, [], [{"throughput_toks": "bad"}], [{"throughput_toks": float("nan")}], [{"throughput_toks": {"bad": 1}}]],
)
def test_bad_rollup_is_unsuccessful(slurm_system: SlurmSystem, tmp_path: Path, runs):
    tdef = make_tdef(tmp_path)
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)
    tdef.job_id_path(tr).write_text("123")
    rollup = tdef.rollup_path(tr)
    assert rollup is not None
    rollup.parent.mkdir(parents=True)
    rollup.write_text(json.dumps({"runs": runs}))
    assert not tdef.was_run_successful(tr).is_successful
    assert SRTSlurmReportGenerationStrategy(slurm_system, tr).get_metric("default") == METRIC_ERROR


def test_agentx_requires_job_local_profiles_and_preserves_concurrencies(slurm_system: SlurmSystem, tmp_path: Path):
    tdef = make_tdef(tmp_path)
    tdef.cmd_args.result_format = "agentx"
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)
    tdef.job_id_path(tr).write_text("123")
    logs = tdef.logs_path(tr)
    assert logs is not None
    output = logs / "agentic"
    output.mkdir(parents=True)
    for concurrency, tput, interactivity in [(1, 5484, 518.1), (48, 88086, 42.8)]:
        data = {
            "conc": concurrency,
            "request_metrics": {
                "throughput": {"per_gpu": {"total_tput_tps": tput}, "output": {"tokens_per_second": tput * 2}},
                "latency": {"full_response_itl": {"p90": 1 / interactivity}, "itl": {"p90": 999}},
            },
        }
        (output / f"result_{concurrency}.json").write_text(json.dumps(data))
        assert agentx_metrics(data)["p90_intvty"] == pytest.approx(interactivity)
    assert not tdef.was_run_successful(tr).is_successful
    for concurrency in (1, 48):
        profile = output / f"conc_{concurrency}" / "aiperf_artifacts" / "profile_export.jsonl"
        profile.parent.mkdir(parents=True)
        profile.write_text('{"metrics": {}}\n')
    assert tdef.was_run_successful(tr).is_successful
    assert len(load_results(tdef, tr)) == 2
    report = SRTSlurmReportGenerationStrategy(slurm_system, tr)
    assert report.get_metric("tput_per_gpu") == 88086
    assert report.get_metric("default") == 176172
    report.generate_report()
    with (tmp_path / "srt_slurm_results.csv").open() as result:
        rows = list(csv.DictReader(result))
    assert {float(row["concurrency"]) for row in rows} == {1, 48}
