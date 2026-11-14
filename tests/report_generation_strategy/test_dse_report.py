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


import pytest
import toml

from cloudai.configurator import Trajectory, TrajectoryRecord
from cloudai.core import CommandGenStrategy, TestRun
from cloudai.models.scenario import TestRunDetails
from cloudai.report_generator.dse_report import build_dse_summaries
from cloudai.systems.slurm.slurm_metadata import SlurmJobMetadata
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition


@pytest.fixture
def scalar_observation_case(slurm_system: SlurmSystem) -> tuple[TestRun, TestRun]:
    case = TestRun(
        name="decoded-observation",
        test=NCCLTestDefinition(
            name="nccl",
            description="NCCL case",
            test_template_name="NcclTest",
            cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl", ngpus=[1, 2]),
            agent_steps=1,
        ),
        num_nodes=1,
        nodes=["node1"],
        iterations=1,
    )

    iteration = slurm_system.output_path / case.name / "0"
    trajectory = Trajectory(iteration_dir=iteration)
    trajectory.append(step=1, action={"ngpus": 2}, reward=1.0, observation=7.5)

    step_dir = iteration / "1"
    step_dir.mkdir()
    step_run = case.apply_params_set({"ngpus": 2})
    step_run.current_iteration = 0
    step_run.step = 1
    step_run.output_path = step_dir
    (step_dir / "stdout.txt").write_text("# Out of bounds values# Avg bus bandwidth")

    details = TestRunDetails.from_test_run(step_run, "", "")
    (step_dir / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).write_text(toml.dumps(details.model_dump(mode="json")))

    job = SlurmJobMetadata(
        job_id=1,
        name=case.name,
        state="COMPLETED",
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        elapsed_time_sec=60,
        exit_code="0:0",
        srun_cmd="srun test",
        test_cmd="test",
        job_root=step_dir,
        job_steps=[],
    )
    (step_dir / "slurm-job.toml").write_text(toml.dumps(job.model_dump(mode="json")))

    return case, step_run


@pytest.mark.parametrize("dump_conflict", [False, True])
def test_dse_summary_receives_decoded_observations_and_actions(
    slurm_system: SlurmSystem, scalar_observation_case: tuple[TestRun, TestRun], dump_conflict: bool
) -> None:
    case, completed = scalar_observation_case
    if dump_conflict:
        from cloudai.configurator import TrajectoryLoadError

        dump = completed.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME
        recorded = toml.loads(dump.read_text())
        recorded["test_definition"]["cmd_args"]["ngpus"] = 1
        dump.write_text(toml.dumps(recorded))
        with pytest.raises(TrajectoryLoadError):
            build_dse_summaries(
                system=slurm_system,
                results_root=slurm_system.output_path,
                loaded_test_runs=[completed],
                test_cases=[case],
            )
        return

    summaries = build_dse_summaries(
        system=slurm_system,
        results_root=slurm_system.output_path,
        loaded_test_runs=[completed],
        test_cases=[case],
    )

    assert len(summaries) == 1
    assert summaries[0].reward_chart_data == {
        "labels": [1],
        "rewards": [1.0],
        "observations": ["7.5"],
        "best_index": 0,
    }
    assert [
        (row.name, [(value.text, value.is_best) for value in row.values]) for row in summaries[0].parameter_rows
    ] == [("ngpus", [("1", False), ("2", True)])]


def test_dse_summary_uses_the_record_observation_formatter(
    slurm_system: SlurmSystem,
    scalar_observation_case: tuple[TestRun, TestRun],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def format_observation(record: TrajectoryRecord) -> str:
        assert record["observation"] == 7.5
        return "record-owned observation"

    monkeypatch.setattr(TrajectoryRecord, "format_observation", format_observation, raising=False)
    case, completed = scalar_observation_case

    summaries = build_dse_summaries(
        system=slurm_system,
        results_root=slurm_system.output_path,
        loaded_test_runs=[completed],
        test_cases=[case],
    )

    assert len(summaries) == 1
    chart = summaries[0].reward_chart_data
    assert chart is not None
    assert chart["observations"] == ["record-owned observation"]


@pytest.mark.parametrize("history", ["absent", "metadata-only", "partial", "corrupt-marker"])
def test_dse_summary_skips_only_truly_absent_history(
    slurm_system: SlurmSystem, scalar_observation_case: tuple[TestRun, TestRun], history: str
) -> None:
    case, completed = scalar_observation_case
    iteration = completed.output_path.parent
    if history in ("absent", "metadata-only"):
        (iteration / "trajectory.csv").unlink()
        (iteration / ".trajectory.commit.json").unlink()
        if history == "absent":
            (iteration / "metadata.csv").unlink()
    elif history == "partial":
        (iteration / "trajectory.csv").write_text("step,action,reward,observation\n1,")
    else:
        (iteration / ".trajectory.commit.json").write_text("{")

    if history == "absent":
        assert (
            build_dse_summaries(
                system=slurm_system,
                results_root=slurm_system.output_path,
                loaded_test_runs=[completed],
                test_cases=[case],
            )
            == []
        )
    else:
        with pytest.raises(RuntimeError, match=r"trajectory|metadata|commit"):
            build_dse_summaries(
                system=slurm_system,
                results_root=slurm_system.output_path,
                loaded_test_runs=[completed],
                test_cases=[case],
            )
