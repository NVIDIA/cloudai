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
from cloudai.workloads.srt_slurm import SRTSlurmReportGenerationStrategy, SRTSlurmTestDefinition
from cloudai.workloads.srt_slurm.results import (
    JobGPUCounts,
    agentx_metrics,
    load_gpu_counts,
    load_results,
    sa_bench_metrics,
)

from .test_srt_slurm import make_tdef, write_job_metadata


def test_fixed_sequence_uses_serving_gpu_counts(tmp_path: Path):
    gpus = JobGPUCounts(aggregate=0, prefill=24, decode=8)
    metrics = sa_bench_metrics(
        {"total_token_throughput": 300583, "output_throughput": 32000, "median_tpot_ms": 44.59, "median_ttft_ms": 1000},
        gpus,
    )
    assert metrics["tput_per_gpu"] == pytest.approx(9393.21875)
    assert metrics["output_tput_per_gpu"] == 4000
    assert metrics["input_tput_per_gpu"] == pytest.approx((300583 - 32000) / 24)
    assert metrics["median_intvty"] == pytest.approx(1000 / 44.59)
    assert metrics["median_ttft"] == 1


def test_agentx_recomputes_normalization_from_job_counts():
    data = {
        "conc": 1,
        "request_metrics": {
            "throughput": {
                "per_gpu": {},
                "total": {"tokens_per_second": 14407},
                "output": {"tokens_per_second": 112},
            },
            "latency": {"full_response_itl": {"p90": 0.00554}, "tpot": {"p90": 0.006}},
        },
    }
    metrics = agentx_metrics(data, JobGPUCounts(aggregate=8, prefill=0, decode=0))
    assert metrics["tput_per_gpu"] == 14407 / 8
    assert metrics["output_tput_per_gpu"] == 14
    assert metrics["p90_intvty"] == pytest.approx(1 / 0.00554)
    data["request_metrics"]["throughput"]["per_gpu"] = {"total_tput_tps": 1801}
    assert agentx_metrics(data, JobGPUCounts(aggregate=16, prefill=0, decode=0))["tput_per_gpu"] == 14407 / 16


@pytest.mark.parametrize(
    "runs",
    [None, [], [{"throughput_toks": "bad"}], [{"throughput_toks": float("nan")}], [{"throughput_toks": {"bad": 1}}]],
)
def test_bad_rollup_is_unsuccessful(slurm_system: SlurmSystem, tmp_path: Path, runs):
    tdef = make_tdef(tmp_path)
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)
    tdef.job_id_path(tr).write_text("123")
    write_job_metadata(tdef, tr)
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
    write_job_metadata(tdef, tr)
    logs = tdef.logs_path(tr)
    assert logs is not None
    output = logs / "agentic"
    output.mkdir(parents=True)
    for concurrency, tput, interactivity in [(1, 5484, 518.1), (48, 88086, 42.8)]:
        data = {
            "conc": concurrency,
            "request_metrics": {
                "throughput": {
                    "per_gpu": {"total_tput_tps": 999999},
                    "total": {"tokens_per_second": tput * 4},
                    "output": {"tokens_per_second": tput * 2},
                },
                "latency": {"full_response_itl": {"p90": 1 / interactivity}, "itl": {"p90": 999}},
            },
        }
        (output / f"result_{concurrency}.json").write_text(json.dumps(data))
        assert agentx_metrics(data, JobGPUCounts(aggregate=4, prefill=0, decode=0))["p90_intvty"] == pytest.approx(
            interactivity
        )
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


@pytest.mark.parametrize(
    "resources, expected",
    [
        ({"agg_workers": 1, "gpus_per_agg": 1}, JobGPUCounts(1, 0, 0)),
        ({"agg_workers": 2, "gpus_per_agg": 2}, JobGPUCounts(4, 0, 0)),
        ({"agg_workers": 1, "gpus_per_agg": 16, "agg_nodes": 2}, JobGPUCounts(16, 0, 0)),
        (
            {"agg_workers": 0, "prefill_workers": 3, "gpus_per_prefill": 8, "decode_workers": 1, "gpus_per_decode": 8},
            JobGPUCounts(0, 24, 8),
        ),
        (
            {
                "agg_workers": 0,
                "prefill_workers": 1,
                "gpus_per_prefill": 4,
                "decode_workers": 1,
                "gpus_per_decode": 4,
                "decode_nodes": 0,
            },
            JobGPUCounts(0, 4, 4),
        ),
    ],
)
def test_job_metadata_resolves_worker_gpu_totals(tmp_path: Path, resources: dict, expected: JobGPUCounts):
    tdef = make_tdef(tmp_path)
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)
    tdef.job_id_path(tr).write_text("123")
    metadata = write_job_metadata(tdef, tr, **resources)
    assert load_gpu_counts(metadata) == expected


@pytest.mark.parametrize("agentic", [False, True])
def test_missing_metadata_fails_even_with_results(slurm_system: SlurmSystem, tmp_path: Path, agentic: bool):
    tdef = make_tdef(tmp_path)
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)
    tdef.job_id_path(tr).write_text("123")
    rollup = tdef.rollup_path(tr)
    assert rollup is not None
    rollup.parent.mkdir(parents=True)
    rollup.write_text(json.dumps({"runs": [{"throughput_toks": 100, "tput_per_gpu": 50}]}))
    if agentic:
        tdef.cmd_args.result_format = "agentx"
        output = rollup.parent / "agentic"
        output.mkdir()
        (output / "result.json").write_text(
            json.dumps(
                {
                    "conc": 1,
                    "request_metrics": {
                        "throughput": {"total": {"tokens_per_second": 100}, "per_gpu": {"total_tput_tps": 50}}
                    },
                }
            )
        )
        profile = output / "conc_1/aiperf_artifacts/profile_export.jsonl"
        profile.parent.mkdir(parents=True)
        profile.write_text("{}\n")
    with pytest.raises(ValueError, match="Missing required srt-slurm job metadata"):
        load_results(tdef, tr)
    status = tdef.was_run_successful(tr)
    assert not status.is_successful
    assert "123.json" in status.error_message
    report = SRTSlurmReportGenerationStrategy(slurm_system, tr)
    assert not report.can_handle_directory()
    assert report.get_metric("tput_per_gpu") == METRIC_ERROR


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        {"job_id": "999", "resources": {}},
        {"job_id": "123"},
        {"job_id": "123", "resources": {"agg_workers": None}},
        {"job_id": "123", "resources": {"gpus_per_agg": None}},
        {"job_id": "123", "resources": {"gpus_per_agg": True}},
        {"job_id": "123", "resources": {"agg_workers": True}},
        {"job_id": "123", "resources": {"agg_workers": -1}},
        {"job_id": "123", "resources": {"agg_workers": 1, "gpus_per_agg": 0}},
        {"job_id": "123", "resources": {"agg_workers": 1, "gpus_per_agg": 1.5}},
        {"job_id": "123", "resources": {"agg_workers": 1, "gpus_per_agg": "4"}},
        {"job_id": "123", "resources": {"agg_workers": 0, "prefill_workers": 0, "decode_workers": 0}},
    ],
)
def test_invalid_metadata_is_unsuccessful(slurm_system: SlurmSystem, tmp_path: Path, invalid):
    tdef = make_tdef(tmp_path)
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)
    tdef.job_id_path(tr).write_text("123")
    path = write_job_metadata(tdef, tr)
    if isinstance(invalid, dict) and isinstance(invalid.get("resources"), dict):
        complete = json.loads(path.read_text())
        complete["resources"].update(invalid["resources"])
        invalid = {**invalid, "resources": complete["resources"]}
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError):
        load_results(tdef, tr)
    assert not tdef.was_run_successful(tr).is_successful
    assert not SRTSlurmReportGenerationStrategy(slurm_system, tr).can_handle_directory()


def test_malformed_metadata_is_unsuccessful(tmp_path: Path):
    tdef = make_tdef(tmp_path)
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)
    tdef.job_id_path(tr).write_text("123")
    path = write_job_metadata(tdef, tr)
    path.write_text('{"job_id":')
    status = tdef.was_run_successful(tr)
    assert not status.is_successful
    assert "job_id" in status.error_message


@pytest.mark.parametrize("workers, per_worker", [(1, 1), (2, 2), (1, 16)])
def test_dse_reporting_uses_each_jobs_metadata(tmp_path: Path, workers: int, per_worker: int):
    tdef = make_tdef(tmp_path)
    base = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path)
    tr = base.apply_params_set(
        {"recipe_overrides.roles.agg.workers": workers, "recipe_overrides.roles.agg.gpus": per_worker}
    )
    assert isinstance(tr.test, SRTSlurmTestDefinition)
    tr.output_path = tmp_path / "selected"
    tr.output_path.mkdir()
    tr.test.job_id_path(tr).write_text("123")
    write_job_metadata(tr.test, tr, agg_workers=workers, gpus_per_agg=per_worker)
    logs = tr.test.logs_path(tr)
    assert logs is not None
    raw = logs / "sa-bench_one" / "results_concurrency_1_gpus_999.json"
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps({"total_token_throughput": 1600, "output_throughput": 800, "max_concurrency": 1}))
    result = load_results(tr.test, tr)[0]
    assert result["serving_gpus"] == workers * per_worker
    assert result["tput_per_gpu"] == 1600 / (workers * per_worker)
    assert tdef.cmd_args.recipe_overrides == {"roles": {"agg": {"gpus": 4}}}


def test_agentx_disaggregated_metrics_use_total_pool():
    data = {
        "conc": 1,
        "request_metrics": {
            "throughput": {
                "total": {"tokens_per_second": 1000},
                "input": {"tokens_per_second": 800},
                "output": {"tokens_per_second": 200},
                "per_gpu": {"total_tput_tps": 999, "input_tput_tps": 999, "output_tput_tps": 999},
            }
        },
    }
    metrics = agentx_metrics(data, JobGPUCounts(aggregate=0, prefill=8, decode=4))
    assert metrics["tput_per_gpu"] == 1000 / 12
    assert metrics["input_tput_per_gpu"] == 800 / 12
    assert metrics["output_tput_per_gpu"] == 200 / 12


def test_agentx_without_raw_total_is_unusable():
    data = {"conc": 1, "request_metrics": {"throughput": {"per_gpu": {"total_tput_tps": 100}}}}
    assert not agentx_metrics(data, JobGPUCounts(aggregate=4, prefill=0, decode=0))


def test_rollup_normalization_is_replaced_and_missing_total_stays_unavailable():
    gpus = JobGPUCounts(aggregate=2, prefill=0, decode=0)
    data = {
        "throughput_toks": 200,
        "total_token_throughput": 1000,
        "tput_per_gpu": 125,
        "input_tput_per_gpu": 100,
        "output_tput_per_gpu": 25,
    }
    metrics = sa_bench_metrics(data, gpus)
    assert metrics["tput_per_gpu"] == 500
    assert metrics["input_tput_per_gpu"] == 400
    assert metrics["output_tput_per_gpu"] == 100
    data.pop("total_token_throughput")
    assert "tput_per_gpu" not in sa_bench_metrics(data, gpus)
