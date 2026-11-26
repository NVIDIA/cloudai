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


import csv
from pathlib import Path

import pandas as pd
import pytest
import toml

import cloudai.configurator as configurator
from cloudai.configurator import Trajectory
from cloudai.core import CommandGenStrategy, TestRun
from cloudai.models.scenario import TestRunDetails
from cloudai.report_generator.dse_report import build_dse_summaries
from cloudai.systems.slurm.slurm_metadata import SlurmJobMetadata
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition
from tests.conftest import selected_dump_cases


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


@pytest.mark.parametrize("dump_conflict", [False, True, "nonwinning"])
def test_dse_summary_receives_decoded_observations_and_actions(
    slurm_system: SlurmSystem, scalar_observation_case: tuple[TestRun, TestRun], dump_conflict: bool | str
) -> None:
    case, completed = scalar_observation_case
    if dump_conflict:
        from cloudai.configurator import TrajectoryLoadError

        selected = completed
        if dump_conflict == "nonwinning":
            history = Trajectory(iteration_dir=completed.output_path.parent, mode="resume")
            history.append(step=2, action={"ngpus": 1}, reward=0.0, observation=7.5)
            selected = case.apply_params_set({"ngpus": 1})
            selected.step = 2
            selected.output_path = completed.output_path.parent / "2"
            selected.output_path.mkdir()
            (selected.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).write_text(
                toml.dumps(TestRunDetails.from_test_run(selected, "", "").model_dump(mode="json"))
            )
        dump = selected.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME
        recorded = toml.loads(dump.read_text())
        recorded["test_definition"]["cmd_args"]["ngpus"] = 1 if dump_conflict is True else 2
        dump.write_text(toml.dumps(recorded))
        before = {path: path.read_bytes() for path in dump.parent.parent.rglob("*") if path.is_file()}
        with pytest.raises(TrajectoryLoadError):
            build_dse_summaries(
                system=slurm_system,
                results_root=slurm_system.output_path,
                loaded_test_runs=[completed],
                test_cases=[case],
            )
        assert {path: path.read_bytes() for path in dump.parent.parent.rglob("*") if path.is_file()} == before
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


@pytest.mark.parametrize(
    ("observation", "present", "expected"),
    [
        (None, False, "n/a"),
        ([], True, "n/a"),
        ("[truncated observation]", True, "n/a"),
        ([1.23456, 2.0, "ready", 0.00001], True, "1.2346, 2, ready, 0"),
        (0, True, "0"),
    ],
)
def test_summary_formats_recorded_observations_without_changing_domains(
    slurm_system: SlurmSystem, scalar_observation_case, tmp_path, observation, present, expected
) -> None:
    case, completed = scalar_observation_case
    results_root = tmp_path / "formatted"
    iteration = results_root / case.name / "0"
    row = {"step": 1, "action.ngpus": 2, "reward": 1.0}
    if present:
        row["observation"] = observation
    history = Trajectory(iteration_dir=iteration, dataframe=pd.DataFrame([row], dtype=object))
    history.append(step=2, action={"ngpus": 2}, reward=2.0, observation=7.5)
    runs = []
    for step in (1, 2):
        selected = case.apply_params_set({"ngpus": 2})
        selected.step = step
        selected.output_path = iteration / str(step)
        selected.output_path.mkdir(parents=True)
        (selected.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).write_text(
            toml.dumps(TestRunDetails.from_test_run(selected, "", "").model_dump(mode="json"))
        )
        timing = toml.loads((completed.output_path / "slurm-job.toml").read_text())
        timing["job_root"] = str(selected.output_path)
        (selected.output_path / "slurm-job.toml").write_text(
            toml.dumps(SlurmJobMetadata.model_validate(timing).model_dump(mode="json"))
        )
        runs.append(selected)
    record = history.records()[0]
    before = repr(record["observation"]) if present else None
    original_files = {path: path.read_bytes() for path in iteration.rglob("*") if path.is_file()}

    summaries = build_dse_summaries(
        system=slurm_system,
        results_root=results_root,
        loaded_test_runs=runs,
        test_cases=[case],
        histories={iteration: history},
    )

    assert len(summaries) == 1
    chart = summaries[0].reward_chart_data
    assert chart is not None
    assert chart["observations"] == [expected, "7.5"]
    assert chart["labels"] == [1, 2] and chart["best_index"] == 1
    assert toml.loads(summaries[0].best_config_toml)["cmd_args"]["ngpus"] == 2
    if present:
        assert repr(record["observation"]) == before
    else:
        with pytest.raises(KeyError):
            _ = record["observation"]
        assert "observation_unusable" in record.ineligibility_reasons("measurement")
    assert {path: path.read_bytes() for path in iteration.rglob("*") if path.is_file()} == original_files


@pytest.mark.parametrize("history", ["absent", "metadata-only", "partial", "corrupt-marker"])
def test_dse_summary_skips_only_truly_absent_history(
    slurm_system: SlurmSystem, scalar_observation_case: tuple[TestRun, TestRun], history: str
) -> None:
    from cloudai.configurator import TrajectoryLoadError

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
        with pytest.raises(TrajectoryLoadError, match=r"trajectory|metadata|commit"):
            build_dse_summaries(
                system=slurm_system,
                results_root=slurm_system.output_path,
                loaded_test_runs=[completed],
                test_cases=[case],
            )


@pytest.mark.parametrize(
    "archive,change",
    selected_dump_cases(),
)
def test_best_config_export_verifies_recorded_settings_without_inventing_defaults(
    slurm_system: SlurmSystem, archive: str, change: str, selected_dump_case
) -> None:
    from cloudai.configurator import Trajectory, TrajectoryLoadError

    selected, case, iteration, expected, dump, export, previous, before, read_attempts, change, valid = (
        selected_dump_case(archive, change)
    )
    if valid:
        summaries = build_dse_summaries(slurm_system, slurm_system.output_path, [selected], [case])
        assert len(summaries) == 1
        actual = toml.loads(summaries[0].best_config_toml)
        assert actual == expected
        assert change != "minimal" or NCCLTestDefinition.model_validate(actual).cmd_args.nthreads == 1
        assert export.read_bytes() == previous
        if change == "credentials":
            exported = type(selected.test).model_validate(actual)
            projected = exported.model_dump(mode="json", context={"purpose": "trajectory"})
            original_projection = selected.test.model_dump(mode="json", context={"purpose": "trajectory"})
            assert toml.loads(toml.dumps(projected)) == toml.loads(toml.dumps(original_projection))
            assert selected.test.cmd_args.hf_token == "synthetic-export-hf"
            assert selected.test.cmd_args.wandb_key == "synthetic-export-wandb"
            if archive == "native":
                stored = Trajectory.load(iteration).records()[0].get("env_params", decode_numbers=True)
                assert {key: projected["cmd_args"][key] for key in ("hf_token", "wandb_key", "alias")} == {
                    key: stored[key] for key in ("hf_token", "wandb_key", "alias")
                }
    else:
        with pytest.raises(TrajectoryLoadError):
            build_dse_summaries(slurm_system, slurm_system.output_path, [selected], [case])
        assert export.read_bytes() == previous
    assert change != "read_error" or read_attempts == [dump]
    assert all(path.read_bytes() == original for path, original in before.items())


@pytest.mark.parametrize("outcome", ["mixed", "all_failed", "no_run", "job_failed"])
def test_best_config_uses_eligible_measurements_without_removing_history(
    slurm_system: SlurmSystem, outcome: str
) -> None:
    from cloudai.configurator import Trajectory
    from cloudai.core import RewardOverrides
    from cloudai.models.output import Metric, Run

    case = TestRun(
        name="measurement-selection",
        test=NCCLTestDefinition(
            name="nccl",
            description="measurement selection",
            test_template_name="NcclTest",
            cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl", ngpus=[1, 2]),
            agent_steps=3,
        ),
        num_nodes=1,
        nodes=["node1"],
    )
    iteration = slurm_system.output_path / case.name / "0"
    history = Trajectory(iteration_dir=iteration)
    runs = []
    for step, ngpus, reward in ((1, 1, -1.0), (2, 2, 100.0), (3, 1, -1.0)):
        selected = case.apply_params_set({"ngpus": ngpus})
        selected.step = step
        selected.output_path = iteration / str(step)
        selected.output_path.mkdir(parents=True)
        (selected.output_path / "stdout.txt").write_text("# Out of bounds values# Avg bus bandwidth")
        job = SlurmJobMetadata(
            job_id=step,
            name=case.name,
            state="FAILED" if outcome == "job_failed" else "COMPLETED",
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:01:00",
            elapsed_time_sec=60,
            exit_code="1:0" if outcome == "job_failed" else "0:0",
            srun_cmd="srun test",
            test_cmd="test",
            job_root=selected.output_path,
            job_steps=[],
        )
        (selected.output_path / "slurm-job.toml").write_text(toml.dumps(job.model_dump(mode="json")))
        (selected.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).write_text(
            toml.dumps(TestRunDetails.from_test_run(selected, "", "").model_dump(mode="json"))
        )
        cached = history.find(action={"ngpus": 1}) if step == 3 else None
        history.record(
            test_run=selected,
            action_space=case.param_space,
            action={"ngpus": ngpus},
            reward=reward,
            observation={"default": reward},
            system=slurm_system,
            rewards=RewardOverrides(),
            execution_error="failed after completed output" if step == 2 or outcome == "all_failed" else None,
            cached_result=cached,
            run_output=None
            if outcome == "no_run" or step == 3
            else Run(
                path=str(selected.output_path),
                jobid=f"job-{step}",
                status="failed" if outcome == "job_failed" else "completed",
                step=step,
                iteration=0,
                metrics=[Metric(name="default", value=reward, unit="unknown")],
            ),
        )
        runs.append(selected)
    records = Trajectory.load(iteration).records()
    assert [record.step for record in records] == [1, 2, 3]
    assert [not record.ineligibility_reasons("measurement") for record in records] == (
        [True, False, True] if outcome in ("mixed", "no_run") else [False, False, False]
    )
    assert not any(record.ineligibility_reasons("decision") for record in records)
    assert records[2].get("measurement", decode_numbers=True)["kind"] == "cached"
    assert (
        records[2].get("measurement", decode_numbers=True)["origin"]
        == records[0].get("measurement", decode_numbers=True)["origin"]
    )
    export = iteration / f"{case.name}.toml"
    previous = b'preserve = "existing export"\n'
    export.write_bytes(previous)
    before = {path: path.read_bytes() for path in iteration.rglob("*") if path.is_file() and path != export}
    summaries = build_dse_summaries(slurm_system, slurm_system.output_path, [] if outcome == "no_run" else runs, [case])
    assert export.read_bytes() == previous
    if outcome in ("mixed", "no_run"):
        assert len(summaries) == 1
        summary = summaries[0]
        assert toml.loads(summary.best_config_toml)["cmd_args"]["ngpus"] == 1
        assert summary.reward_chart_data == {
            "labels": [1, 2, 3],
            "rewards": [-1.0, 100.0, -1.0],
            "observations": ["-1", "100", "-1"],
            "best_index": 0,
        }
        assert summary.efficiency_steps == "3 / 2 steps"
        assert summary.observed_runtime == "3m"
        assert [
            (row.name, [(value.text, value.is_best) for value in row.values]) for row in summary.parameter_rows
        ] == [("ngpus", [("1", True), ("2", False)])]
    else:
        assert summaries == []
    assert {path: path.read_bytes() for path in iteration.rglob("*") if path.is_file() and path != export} == before
    assert len(Trajectory.load(iteration).records()) == 3


@pytest.mark.parametrize(
    "archive", ["native_partial", "native_full", "legacy_literal", "unversioned_opaque_literal", "declared_incomplete"]
)
def test_parameter_highlights_use_complete_recorded_actions_with_raw_fallback(
    slurm_system: SlurmSystem, archive: str
) -> None:
    from cloudai.core import RewardOverrides
    from cloudai.models.output import Metric, Run

    case = TestRun(
        name="complete-highlights",
        test=NCCLTestDefinition(
            name="nccl",
            description="complete highlights",
            test_template_name="NcclTest",
            cmd_args=NCCLCmdArgs.model_validate(
                {"docker_image_url": "fake://url/nccl", "ngpus": [1, 2], "label": ["001", "002"]}
            ),
            agent_steps=1,
        ),
        num_nodes=1,
        nodes=["node1"],
    )
    assert list(case.param_space) == ["ngpus", "label"]
    selected = case.apply_params_set({"ngpus": 2, "label": "001"})
    selected.step = 1
    iteration = slurm_system.output_path / case.name / "0"
    selected.output_path = iteration / "1"
    selected.output_path.mkdir(parents=True)
    (selected.output_path / "stdout.txt").write_text("# Out of bounds values# Avg bus bandwidth")
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
        job_root=selected.output_path,
        job_steps=[],
    )
    (selected.output_path / "slurm-job.toml").write_text(toml.dumps(job.model_dump(mode="json")))
    (selected.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).write_text(
        toml.dumps(TestRunDetails.from_test_run(selected, "", "").model_dump(mode="json"))
    )
    history = Trajectory(iteration_dir=iteration)
    action = {"ngpus": 2} if archive == "native_partial" else {"ngpus": 2, "label": "001"}
    if archive in ("legacy_literal", "unversioned_opaque_literal"):
        context = (
            {"run_details": {"resolved_action": {"label": "001"}}} if archive == "unversioned_opaque_literal" else {}
        )
        history.append(step=1, action=action, reward=9, observation=[1.0], **context)
    else:
        history.record(
            test_run=selected,
            action_space={"ngpus": case.param_space["ngpus"]} if archive == "declared_incomplete" else case.param_space,
            action=action,
            reward=9,
            observation={"default": 1.0},
            system=slurm_system,
            rewards=RewardOverrides(),
            run_output=Run(
                path=str(selected.output_path),
                jobid="highlight",
                status="completed",
                step=1,
                iteration=0,
                metrics=[Metric(name="default", value=1.0, unit="unknown")],
            ),
        )
    before = {path: path.read_bytes() for path in iteration.rglob("*") if path.is_file()}
    original = history.records()[0]
    original_action = original["action"]
    original_metadata = original.metadata
    summaries = build_dse_summaries(slurm_system, slurm_system.output_path, [selected], [case])
    assert len(summaries) == 1
    actual = toml.loads(summaries[0].best_config_toml)["cmd_args"]
    assert {name: actual[name] for name in case.param_space} == {"ngpus": 2, "label": "001"}
    assert type(actual["label"]) is str
    assert original["action"] == original_action == action
    assert original.metadata == original_metadata
    assert {path: path.read_bytes() for path in iteration.rglob("*") if path.is_file()} == before
    assert [
        (row.name, [(value.text, value.is_best) for value in row.values]) for row in summaries[0].parameter_rows
    ] == [
        ("ngpus", [("1", False), ("2", True)]),
        ("label", [("001", True), ("002", False)]),
    ]


@pytest.mark.parametrize("archive", ["legacy", "native"])
@pytest.mark.parametrize(
    "rewards,expected_step,node_only,layout",
    [
        pytest.param((9007199254740992, 9007199254740993), 2, False, "complete", id="adjacent_native_integers"),
        pytest.param(("9007199254740992", " 9007199254740993"), 2, False, "complete", id="adjacent_integer_strings"),
        pytest.param((9007199254740993, 9007199254740993), 1, False, "complete", id="exact_bigint_tie"),
        pytest.param((9, "010"), 2, False, "complete", id="mixed_integer_string"),
        pytest.param(("09", "010"), 2, False, "complete", id="leading_zero_strings"),
        pytest.param((9, " 10.5"), 2, False, "complete", id="fraction_string"),
        pytest.param((9.5, 10.5), 2, False, "complete", id="native_floats"),
        pytest.param((10, "010"), 1, False, "complete", id="numeric_tie"),
        pytest.param((1, 2), 2, True, "complete", id="node_only"),
        pytest.param((True, 2), 2, False, "complete", id="unusable_boolean_evidence"),
        pytest.param(("N/A", 10), 2, False, "complete", id="unusable_reward"),
        pytest.param(("NaN", 10), 2, False, "complete", id="nan_reward"),
        pytest.param(("Infinity", 10), 2, False, "complete", id="positive_infinity_reward"),
        pytest.param(("-Infinity", 10), 2, False, "complete", id="negative_infinity_reward"),
        pytest.param((1, 2), 2, False, "missing_stdout", id="winner_without_stdout"),
        pytest.param((1, 2), 2, False, "no_loaded_runs", id="winner_without_loaded_runs"),
        pytest.param((5, 10, 4), 2, False, "gaps", id="gapped_positive_steps"),
    ],
)
def test_summary_ranks_exact_rewards_and_highlights_selected_resources(
    slurm_system: SlurmSystem,
    archive: str,
    rewards: tuple[int | float | str, ...],
    expected_step: int,
    node_only: bool,
    layout: str,
) -> None:
    import json
    import math

    from cloudai.core import RewardOverrides
    from cloudai.models.output import Metric, Run

    case = TestRun(
        name="exact-summary",
        test=NCCLTestDefinition(
            name="nccl",
            description="exact reward summary",
            test_template_name="NcclTest",
            cmd_args=NCCLCmdArgs(
                docker_image_url="fake://url/nccl", ngpus=1 if node_only else list(range(1, len(rewards) + 1))
            ),
            agent_steps=len(rewards),
        ),
        num_nodes=[2, 4] if node_only else 1,
        nodes=["node1"],
    )
    dimension = "NUM_NODES" if node_only else "ngpus"
    assert list(case.param_space) == [dimension]
    assert case.is_dse_job
    assert case.test.is_dse_job is not node_only
    iteration = slurm_system.output_path / case.name / "0"
    history = Trajectory(iteration_dir=iteration)
    runs = []
    for ordinal, reward in enumerate(rewards, 1):
        step = ordinal * 2 - 1 if layout == "gaps" else ordinal - 1 if layout == "zero" else ordinal
        value = ordinal * 2 if node_only else ordinal
        action = {dimension: value}
        selected = case.apply_params_set(action)
        selected.step = step
        selected.output_path = iteration / str(step)
        selected.output_path.mkdir(parents=True)
        if layout != "missing_stdout" or ordinal != expected_step:
            (selected.output_path / "stdout.txt").write_text("# Out of bounds values# Avg bus bandwidth")
        job = SlurmJobMetadata(
            job_id=step,
            name=case.name,
            state="COMPLETED",
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:01:00",
            elapsed_time_sec=60,
            exit_code="0:0",
            srun_cmd="srun test",
            test_cmd="test",
            job_root=selected.output_path,
            job_steps=[],
        )
        (selected.output_path / "slurm-job.toml").write_text(toml.dumps(job.model_dump(mode="json")))
        (selected.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).write_text(
            toml.dumps(TestRunDetails.from_test_run(selected, "", "").model_dump(mode="json"))
        )
        if archive == "native":
            history.record(
                test_run=selected,
                action_space=case.param_space,
                action=action,
                reward=reward,
                observation={"default": 1.0},
                system=slurm_system,
                rewards=RewardOverrides(),
                run_output=Run(
                    path=str(selected.output_path),
                    jobid=f"summary-{step}",
                    status="completed",
                    step=step,
                    iteration=0,
                    metrics=[Metric(name="default", value=1.0, unit="unknown")],
                ),
            )
        else:
            history.append(step=step, action=action, reward=reward, observation=[1.0])
        runs.append(selected)
    records = Trajectory.load(iteration).records()
    original = [(record["reward"], type(record["reward"]), record.metadata) for record in records]
    getters = [
        (
            record.get("reward"),
            type(record.get("reward", decode_numbers=True)),
            repr(record.get("reward", decode_numbers=True)),
        )
        for record in records
    ]
    before = {path: path.read_bytes() for path in iteration.rglob("*") if path.is_file()}
    summaries = build_dse_summaries(
        slurm_system, slurm_system.output_path, [] if layout == "no_loaded_runs" else runs, [case]
    )
    assert len(summaries) == 1
    summary = summaries[0]
    chart = summary.reward_chart_data
    assert chart is not None
    assert chart["best_index"] == expected_step - 1
    expected_rewards = []
    for reward in rewards:
        try:
            numeric = (
                reward
                if type(reward) is int
                else int(reward)
                if isinstance(reward, str) and reward.strip().isdigit()
                else float(reward)
            )
        except ValueError:
            numeric = None
        if numeric is not None and not math.isfinite(numeric):
            numeric = None
        expected_rewards.append(numeric)
    assert all(reward is None or type(reward) in (int, float) for reward in chart["rewards"])
    assert chart["rewards"] == expected_rewards
    assert json.loads(json.dumps(chart, allow_nan=False))["rewards"] == expected_rewards
    if expected_rewards[0] is None:
        assert "reward_unusable" in records[0].ineligibility_reasons("measurement")
    expected_value = expected_step * 2 if node_only else expected_step
    assert [(row.name, [(value.text, value.is_best) for value in row.values]) for row in summary.parameter_rows] == [
        (dimension, [(str(value), value == expected_value) for value in case.param_space[dimension]])
    ]
    actual = toml.loads(summary.best_config_toml)
    assert actual["cmd_args"]["ngpus"] == (1 if node_only else expected_step)
    assert set(actual) <= NCCLTestDefinition.model_fields.keys() | {"cmd_args"}
    dump = toml.loads((runs[expected_step - 1].output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).read_text())
    assert dump["nnodes"] == (expected_value if node_only else 1)
    assert {path: path.read_bytes() for path in iteration.rglob("*") if path.is_file()} == before
    reloaded = Trajectory.load(iteration).records()
    assert [(record["reward"], type(record["reward"]), record.metadata) for record in reloaded] == original
    assert [
        (
            record.get("reward"),
            type(record.get("reward", decode_numbers=True)),
            repr(record.get("reward", decode_numbers=True)),
        )
        for record in reloaded
    ] == getters


@pytest.mark.parametrize(
    "rows",
    [
        [[1, {"x": "001"}, "010", []]],
        [[1, {"x": 1}, "", []]],
        [[1, {"x": float("nan")}, "nan", [float("nan")]]],
        [[1, {"x": 1}, "1", []], [2, {"x": 2}, "2.5", []]],
        [[1, {"model": {"layers": 8}, "empty": {}}, "2", [1.0]]],
        [],
    ],
    ids=["leading-zero", "empty-reward", "nan", "mixed-numbers", "nested-actions", "header-only"],
)
def test_summary_preserves_legacy_history_values(
    slurm_system, scalar_observation_case, tmp_path: Path, monkeypatch, rows
):
    case, completed = scalar_observation_case
    results_root = tmp_path / "legacy-report"
    iteration = results_root / case.name / "0"
    iteration.mkdir(parents=True)
    source = iteration / "trajectory.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "action", "reward", "observation"])
        for step, action, reward, observation in rows:
            writer.writerow([step, repr(action), reward, repr(observation)])
    if not rows:
        empty = source.read_bytes()
        assert build_dse_summaries(slurm_system, results_root, [], [case]) == []
        assert source.read_bytes() == empty
    with source.open("a", newline="") as stream:
        csv.writer(stream).writerow([len(rows) + 1, repr({"ngpus": 2}), "100", repr([1.0])])
    selected = case.apply_params_set({"ngpus": 2})
    selected.step = len(rows) + 1
    selected.output_path = iteration / str(selected.step)
    selected.output_path.mkdir()
    (selected.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).write_text(
        toml.dumps(TestRunDetails.from_test_run(selected, "", "").model_dump(mode="json"))
    )
    timing = toml.loads((completed.output_path / "slurm-job.toml").read_text())
    timing["job_root"] = str(selected.output_path)
    (selected.output_path / "slurm-job.toml").write_text(
        toml.dumps(SlurmJobMetadata.model_validate(timing).model_dump(mode="json"))
    )
    original = {path: path.read_bytes() for path in iteration.rglob("*") if path.is_file()}
    snapshots = []
    load = Trajectory.load

    def capture(*args, **kwargs):
        history = load(*args, **kwargs)
        snapshots.append(history)
        return history

    monkeypatch.setattr(Trajectory, "load", capture)
    summaries = build_dse_summaries(slurm_system, results_root, [], [case])

    assert len(summaries) == 1
    chart = summaries[0].reward_chart_data
    assert chart is not None
    assert chart["labels"] == list(range(1, len(rows) + 2))
    assert chart["rewards"] == [None if reward in ("", "nan") else float(reward) for _, _, reward, _ in rows] + [100]
    assert chart["best_index"] == len(rows)
    assert toml.loads(summaries[0].best_config_toml)["cmd_args"]["ngpus"] == 2
    assert len(snapshots) == 1
    assert snapshots[0].output_path == source
    for record, (_, action, reward, observation) in zip(snapshots[0].records()[: len(rows)], rows, strict=True):
        assert repr(record["action"]) == repr(action)
        assert str(record["reward"]) == reward
        assert repr(record["observation"]) == repr(observation)
    assert {path: path.read_bytes() for path in iteration.rglob("*") if path.is_file()} == original


@pytest.mark.parametrize("artifact", ["absent", "orphan", "duplicate-header", "wide-row", "missing-metadata-step"])
def test_report_reader_uses_shared_integrity_boundary(
    slurm_system, scalar_observation_case, tmp_path: Path, artifact: str
):
    case, _ = scalar_observation_case
    results_root = tmp_path / "strict-report"
    iteration = results_root / case.name / "0"
    iteration.mkdir(parents=True)
    source = iteration / "trajectory.csv"
    if artifact == "orphan":
        (iteration / "metadata.csv").write_text("step,execution.ticket\n1,T-1\n")
    elif artifact == "duplicate-header":
        source.write_text("step,action,reward,reward,observation\n1,{},1,2,[]\n")
    elif artifact == "wide-row":
        source.write_text("step,action,reward,observation\n1,{},1,[],extra\n")
    elif artifact == "missing-metadata-step":
        source.write_text("step,action,reward,observation\n1,{},1,[]\n2,{},2,[]\n")
        (iteration / "metadata.csv").write_text("step,execution.ticket\n1,T-1\n")
    before = {path.name: path.read_bytes() for path in iteration.iterdir()}
    if artifact == "absent":
        assert build_dse_summaries(slurm_system, results_root, [], [case]) == []
    else:
        with pytest.raises(configurator.TrajectoryLoadError):
            build_dse_summaries(slurm_system, results_root, [], [case])
    assert {path.name: path.read_bytes() for path in iteration.iterdir()} == before
