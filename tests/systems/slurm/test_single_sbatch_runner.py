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

from typing import Literal
from unittest.mock import patch

import pytest

import cloudai.models.output
from cloudai.configurator import Trajectory
from cloudai.configurator.env_params import EnvParams, EnvParamSpec
from cloudai.core import TestRun, TestScenario
from cloudai.systems.slurm import SingleSbatchRunner, SlurmSystem


@pytest.mark.parametrize("completion", ["completed", "failed"])
def test_completed_dse_uses_the_shared_recorder_for_full_context(
    dse_tr: TestRun,
    slurm_system: SlurmSystem,
    monkeypatch: pytest.MonkeyPatch,
    completion: Literal["completed", "failed"],
) -> None:
    dse_tr.test.cmd_args.nthreads = [1, 2]
    dse_tr.test.env_params = {"nthreads": EnvParamSpec()}
    dse_tr.test.extra_cmd_args = {"--contract-label": "001"}
    dse_tr.test.agent_metrics = ["bandwidth"]
    dse_tr.test.agent_config = {"random_seed": 42, "rewards": {"metric_failure": -17}}
    runner = SingleSbatchRunner(
        mode="run",
        system=slurm_system,
        test_scenario=TestScenario(name="context", test_runs=[dse_tr]),
        output_path=slurm_system.output_path / "shared-recorder",
    )
    params = EnvParams.from_test(dse_tr.test)
    assert params is not None
    samples = [params.sample(step) for step in range(1, len(dse_tr.all_combinations) + 1)]
    monkeypatch.setattr(TestRun, "get_metric_value", lambda run, system, metric: float(run.step))
    for current in runner.unroll_dse(dse_tr):
        runner.experiment_output.update_run(
            current.name,
            cloudai.models.output.Run(
                path=str(current.output_path),
                jobid=f"job-{current.step}",
                iteration=current.current_iteration,
                step=current.step,
                status=completion,
                metrics=[cloudai.models.output.Metric(name="bandwidth", value=float(current.step), unit="GB/s")],
            ),
        )
    original_record = Trajectory.record
    with patch.object(Trajectory, "record", autospec=True, side_effect=original_record) as record:
        runner.handle_dse()
    assert record.call_count == len(samples)
    assert [call.kwargs["test_run"].test.cmd_args.nthreads for call in record.call_args_list] == [
        sample["nthreads"] for sample in samples
    ]
    assert all(call.kwargs["rewards"].metric_failure == -17 for call in record.call_args_list)
    assert all(call.kwargs["system"] is slurm_system for call in record.call_args_list)
    source = runner.scenario_root / dse_tr.name / str(dse_tr.current_iteration)
    records = Trajectory.load(source).records()
    assert [row.get("env_params", decode_numbers=True)["nthreads"] for row in records] == [
        sample["nthreads"] for sample in samples
    ]
    for row in records:
        assert row.get("env_params", decode_numbers=True)["extra_cmd_args"] == {"--contract-label": "001"}
        assert row.get("env_params", decode_numbers=True)["NUM_NODES"] == dse_tr.num_nodes
        details = row.get("run_details", decode_numbers=True)
        measurement = row.get("measurement", decode_numbers=True)
        assert details["evaluation_policy"]["metric_failure"] == -17
        assert details["step"] == row.step
        assert measurement["kind"] == "fresh"
        assert measurement["execution_status"] == completion
        assert measurement["outcome"] == ("success" if completion == "completed" else "runner_failure")
        assert row.get("evaluation_failure", decode_numbers=True) == (
            None if completion == "completed" else {"constraint": False, "metric_indices": [], "runner": True}
        )
        assert row["reward"] > 0
        assert row.ineligibility_reasons("reference") == ()
        assert ("explicit_failure" in row.ineligibility_reasons("measurement")) == (completion == "failed")
        assert measurement["origin"]["history_id"] == details["history_id"]
        assert measurement["metrics"] == {"0": {"name": "bandwidth", "unit": "GB/s", "status": "observed"}}
