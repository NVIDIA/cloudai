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


@pytest.mark.parametrize(
    "archive,change",
    [
        ("native", "matching"),
        ("native", "minimal"),
        ("native", "command"),
        ("native", "held_action"),
        ("native", "string_action"),
        ("native", "float_action"),
        ("native", "bool_action"),
        ("native", "static"),
        ("native", "resource"),
        ("native", "bool_resource"),
        ("native", "nodes"),
        ("native", "name"),
        ("native", "iteration"),
        ("native", "step"),
        ("native", "bool_step"),
        ("native", "output_path"),
        ("native", "missing_action"),
        ("legacy", "matching"),
        ("legacy", "minimal"),
        ("legacy", "command"),
        ("native", "credentials"),
        ("legacy", "credentials"),
        ("native", "bool_static"),
        ("native", "none_static"),
        ("native", "float_resource"),
        ("native", "string_resource"),
        ("native", "float_step"),
        ("native", "string_step"),
        ("native", "date"),
        ("native", "none_array"),
        ("native", "list_order"),
        ("native", "nested_matching"),
        ("native", "nested_extra"),
        ("native", "nested_missing"),
        ("legacy", "nested_matching"),
        ("legacy", "nested_extra"),
        ("legacy", "nested_missing"),
    ],
)
def test_best_config_export_verifies_recorded_settings_without_inventing_defaults(
    slurm_system: SlurmSystem, archive: str, change: str
) -> None:
    from copy import deepcopy
    from datetime import date

    from cloudai.configurator import Trajectory, TrajectoryLoadError
    from cloudai.core import RewardOverrides
    from cloudai.models.output import Metric, Run

    case = TestRun(
        name="typed-export",
        test=NCCLTestDefinition(
            name="nccl",
            description="typed export",
            test_template_name="NcclTest",
            cmd_args=NCCLCmdArgs.model_validate(
                {
                    "docker_image_url": "fake://url/nccl",
                    **(
                        {"fixed_array": {"none_array": [None], "list_order": [1, 2]}[change]}
                        if change in ("none_array", "list_order")
                        else {}
                    ),
                    **({"date_alias": date(2026, 1, 2)} if change == "date" else {}),
                    "ngpus": [1, 2],
                    "iters": [10, 20],
                    "payload": (
                        [{"x": 1}, {"x": 2}]
                        if change.startswith("nested_")
                        else {"label": "0001", "enabled": True, "missing": None, "a.b": ["001", "true"]}
                    ),
                }
            ),
            dse_excluded_args=([] if change.startswith("nested_") else ["cmd_args.payload"])
            + (["cmd_args.fixed_array"] if change in ("none_array", "list_order") else []),
            agent_steps=1,
            extra_env_vars={"FIXED": "0001"},
        ),
        num_nodes=1,
        nodes=["node1"],
        iterations=1,
    )
    if change == "credentials":
        from cloudai.core import GitRepo
        from cloudai.workloads.megatron_bridge.megatron_bridge import (
            MegatronBridgeCmdArgs,
            MegatronBridgeTestDefinition,
        )

        case.test = MegatronBridgeTestDefinition(
            name="bridge",
            description="credential export",
            test_template_name="MegatronBridge",
            cmd_args=MegatronBridgeCmdArgs.model_validate(
                {
                    "ngpus": [1, 2],
                    "iters": [10, 20],
                    "hf_token": "synthetic-export-hf",
                    "wandb_key": "synthetic-export-wandb",
                    "alias": "synthetic-export-hf:synthetic-export-wandb",
                    "model_family_name": "qwen3",
                    "model_recipe_name": "30b_a3b",
                }
            ),
            git_repos=[GitRepo(url="https://github.com/NVIDIA-NeMo/Megatron-Bridge.git", commit="r0.2.0")],
            agent_steps=1,
        )
    iteration = slurm_system.output_path / case.name / "0"
    selected = case.apply_params_set(
        {"ngpus": 2, "iters": 10} | ({"payload": {"x": 1}} if change.startswith("nested_") else {})
    )
    assert "fixed_array" not in case.param_space
    assert selected.test.cmd_args.model_dump().get("fixed_array") == {"none_array": [None], "list_order": [1, 2]}.get(
        change
    )
    selected.step = 1
    selected.output_path = iteration / "1"
    selected.output_path.mkdir(parents=True)
    (selected.output_path / "stdout.txt").write_text("# Out of bounds values# Avg bus bandwidth")
    (selected.output_path / "cloudai_megatron_bridge_launcher.log").write_text(
        "Step Time : 1.0 s GPU utilization: 2.0 TFLOP/s/GPU"
    )
    (selected.output_path / "cloudai_megatron_bridge_evaluation.exitcode").write_text("0")
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
    history = Trajectory(iteration_dir=iteration)
    if archive == "native":
        history.record(
            test_run=selected,
            action_space=case.param_space,
            action={"ngpus": 2} | ({"payload": {"x": 1}} if change.startswith("nested_") else {}),
            reward=1.0,
            observation={"default": 1.0},
            system=slurm_system,
            rewards=RewardOverrides(),
            run_output=Run(
                path=str(selected.output_path),
                jobid="export",
                status="completed",
                step=1,
                iteration=0,
                metrics=[Metric(name="default", value=1.0, unit="unknown")],
            ),
        )
    else:
        history.append(
            step=1,
            action={"ngpus": 2, "iters": 10} | ({"payload": {"x": 1}} if change.startswith("nested_") else {}),
            reward=1.0,
            observation=1.0,
        )
    raw = toml.loads(toml.dumps(TestRunDetails.from_test_run(selected, "", "").model_dump()))
    if change == "minimal":
        raw["test_definition"] = {
            key: raw["test_definition"][key] for key in ("name", "description", "test_template_name", "cmd_args")
        }
        raw["test_definition"]["cmd_args"] = {
            key: raw["test_definition"]["cmd_args"][key] for key in ("docker_image_url", "ngpus", "iters", "payload")
        }
    updates = {
        "nested_extra": (("test_definition", "cmd_args", "payload", "y"), 2),
        "nested_missing": (("test_definition", "cmd_args", "payload"), {}),
        "none_array": (("test_definition", "cmd_args", "fixed_array"), []),
        "list_order": (("test_definition", "cmd_args", "fixed_array"), [2, 1]),
        "command": (("test_definition", "cmd_args", "ngpus"), 1),
        "held_action": (("test_definition", "cmd_args", "iters"), 999),
        "string_action": (("test_definition", "cmd_args", "iters"), "10"),
        "float_action": (("test_definition", "cmd_args", "iters"), 10.0),
        "bool_action": (("test_definition", "cmd_args", "iters"), True),
        "static": (("test_definition", "extra_env_vars", "FIXED"), "changed"),
        "resource": (("nnodes",), 2),
        "bool_resource": (("nnodes",), True),
        "nodes": (("nodes",), ["other-node"]),
        "name": (("name",), "other-run"),
        "iteration": (("current_iteration",), 1),
        "step": (("step",), 2),
        "bool_step": (("step",), True),
        "output_path": (("output_path",), str(iteration / "other" / "1")),
        "bool_static": (("test_definition", "cmd_args", "payload", "enabled"), 1),
        "none_static": (("test_definition", "cmd_args", "payload", "missing"), "not-none"),
        "float_resource": (("nnodes",), 1.0),
        "string_resource": (("nnodes",), "1"),
        "float_step": (("step",), 1.0),
        "string_step": (("step",), "1"),
    }
    if change in updates:
        path, value = updates[change]
        target = raw
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    elif change == "missing_action":
        del raw["test_definition"]["cmd_args"]["iters"]
    expected = deepcopy(raw["test_definition"])
    dump = selected.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME
    dump.write_text(toml.dumps(raw))
    export = iteration / f"{case.name}.toml"
    previous = b'preserve = "existing export"\n'
    export.write_bytes(previous)
    before = {
        path: path.read_bytes() for path in (history.output_path, history.output_path.with_name("metadata.csv"), dump)
    }
    if change in ("matching", "minimal", "credentials", "date", "nested_matching"):
        summaries = build_dse_summaries(slurm_system, slurm_system.output_path, [selected], [case])
        assert len(summaries) == 1
        actual = toml.loads(summaries[0].best_config_toml)
        assert actual == expected
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
    if outcome == "mixed":
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
