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


from pathlib import Path

import pytest
import toml
from pydantic import Field

from cloudai.core import CmdArgs, SOLMetricModel, TestDefinition, TestRun, TestScenario, TestScenarioParser
from cloudai.models.scenario import ReportConfig
from cloudai.reporter import DSEReporter
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition
from cloudai.workloads.slurm_container import SlurmContainerCmdArgs, SlurmContainerTestDefinition
from cloudai.workloads.slurm_container.slurm_command_gen_strategy import SlurmContainerCommandGenStrategy


def test_scenario_schema_is_complete_without_rebuilding():
    assert TestScenario.__pydantic_complete__
    schema = TestScenario.model_json_schema()
    assert schema["properties"]["Tests"]["items"] == {"$ref": "#/$defs/TestDefinition"}
    assert "ReportConfig" in schema["$defs"]


@pytest.mark.parametrize("case_sol", [None, 42.0, {"bandwidth": [{"value": 120}]}])
def test_one_sol_field_preserves_inheritance_and_snapshot_format(
    definition: NCCLTestDefinition, slurm_system: SlurmSystem, tmp_path: Path, case_sol: float | dict | None
):
    slurm_system.sol = SOLMetricModel.model_validate({"bandwidth": [{"value": 100}], "latency": [{"value": 10}]})
    parser = TestScenarioParser(tmp_path / "scenario.toml", slurm_system, {"nccl": definition}, {})
    case: dict = {"id": "case", "test_name": "nccl", "num_nodes": 1}
    if case_sol is not None:
        case["sol"] = case_sol
    scenario = parser._parse_data({"name": "scenario", "sol": {"bandwidth": [{"value": 110}]}, "Tests": [case]})
    run = scenario.test_runs[0]
    assert run.sol is run.test.sol
    assert not hasattr(run, "metric_sol")
    assert "metric_sol" not in run.test.model_dump()
    payload = toml.loads(toml.dumps(run.model_dump()))
    restored = TestRun.model_validate(payload).test
    if isinstance(case_sol, float):
        assert run.sol == restored.sol == 42.0
    else:
        assert isinstance(run.sol, SOLMetricModel)
        assert isinstance(restored.sol, SOLMetricModel)
        assert run.sol.root["bandwidth"][0].value == (120 if case_sol else 110)
        assert run.sol.root["latency"][0].value == 10
        assert restored.sol == run.sol
        assert "root" not in payload["test"]["sol"]
    assert definition.sol is None
    assert slurm_system.sol.root["bandwidth"][0].value == 100
    assert scenario.sol.root["bandwidth"][0].value == 110


@pytest.fixture
def definition() -> NCCLTestDefinition:
    return NCCLTestDefinition(
        id="case",
        name="nccl",
        description="NCCL",
        test_template_name="NcclTest",
        cmd_args=NCCLCmdArgs(docker_image_url="fake://nccl", ngpus=[1, 2]),
        num_nodes=[1, 2],
        iterations=3,
        pin_nodes=False,
        extra_env_vars={"MODE": ["a", "b"]},
        extra_container_mounts=["/input:/input"],
    )


def test_run_holds_prepared_definition_and_runtime_state_only(definition: NCCLTestDefinition):
    prepared = definition.model_copy(deep=True, update={"num_nodes": 2, "time_limit": "10m", "weight": 25.0})
    run = TestRun(test=prepared)
    assert run.test is prepared
    assert run.nnodes == prepared.num_nodes == 2
    assert run.time_limit == prepared.time_limit == "10m"
    assert run.test.weight == prepared.weight == 25.0
    assert all(not hasattr(run, name) for name in ("weight", "ideal_perf", "pin_nodes"))
    assert set(vars(run)) == {
        "test",
        "nodes",
        "output_path",
        "current_iteration",
        "step",
        "dependencies",
        "pre_test",
        "post_test",
    }
    with pytest.raises(AttributeError):
        setattr(run, "iterations", 10)  # noqa: B010
    assert prepared.iterations == 3
    assert definition.num_nodes == [1, 2]


def test_execution_mutations_are_isolated(definition: NCCLTestDefinition):
    first = TestRun(test=definition.model_copy(deep=True, update={"id": "first"}))
    second = TestRun(test=definition.model_copy(deep=True, update={"id": "second"}))
    first.test.cmd_args.ngpus = 8
    first.test.extra_env_vars["MODE"] = "selected"
    first.test.extra_container_mounts.append("/output:/output")
    first.nodes.append("node1")
    first.test = first.test.model_copy(update={"iterations": 1})
    first.test = first.test.model_copy(update={"pin_nodes": True})
    first.test = first.test.model_copy(update={"extra_srun_args": ["--exclusive"]})
    assert first.test is not definition
    assert type(first.test) is NCCLTestDefinition
    assert second.test.cmd_args.ngpus == definition.cmd_args.ngpus == [1, 2]
    assert second.test.extra_env_vars == definition.extra_env_vars == {"MODE": ["a", "b"]}
    assert second.test.extra_container_mounts == definition.extra_container_mounts == ["/input:/input"]
    assert second.nodes == definition.nodes == []
    assert second.iterations == definition.iterations == 3
    assert not definition.pin_nodes
    assert definition.extra_srun_args == []


def test_materialized_trial_has_one_prepared_definition(definition: NCCLTestDefinition):
    template = TestRun(test=definition.model_copy(deep=True, update={"id": "case"}))
    selected = template.apply_params_set({"ngpus": 2, "NUM_NODES": 2, "extra_env_vars.MODE": "b"})
    assert selected.test is not template.test
    assert selected.test.num_nodes == 2
    assert not hasattr(selected, "definition")
    assert selected.nnodes == 2
    assert selected.test.cmd_args.ngpus == 2
    assert selected.test.extra_env_vars["MODE"] == "b"
    assert selected.param_space == {}
    assert not selected.is_dse_job
    assert template.param_space == {"ngpus": [1, 2], "extra_env_vars.MODE": ["a", "b"], "NUM_NODES": [1, 2]}
    assert definition.cmd_args.ngpus == [1, 2]


def test_scenario_bindings_do_not_replace_definitions(definition: NCCLTestDefinition):
    scenario = TestScenario(name="scenario", tests=[definition])
    run = TestRun(test=definition.model_copy(deep=True, update={"id": "case"}))
    scenario.test_runs = [run]
    scenario.test_runs = [run.apply_params_set({"ngpus": 1, "NUM_NODES": 1, "extra_env_vars.MODE": "a"})]
    assert scenario.tests == [definition]
    assert scenario.tests[0] is definition
    assert "test_runs" not in scenario.model_dump()


def test_parser_resolves_repeated_definitions_independently(slurm_system: SlurmSystem, definition: NCCLTestDefinition):
    parser = TestScenarioParser(Path("scenario.toml"), slurm_system, {"nccl": definition}, {})
    scenario = parser._parse_data(
        {
            "name": "scenario",
            "Tests": [
                {"id": "first", "test_name": "nccl", "num_nodes": 2, "weight": 1},
                {"id": "second", "test_name": "nccl", "iterations": 5, "weight": 3},
            ],
        }
    )
    first, second = scenario.test_runs
    assert first.test is not scenario.tests[0]
    assert second.test is not scenario.tests[1]
    assert [test.id for test in scenario.tests] == ["first", "second"]
    assert [test.weight for test in scenario.tests] == [1, 3]
    assert [first.test.weight, second.test.weight] == [25, 75]
    first.test.extra_env_vars["MODE"] = "a"
    first.test = first.test.model_copy(update={"iterations": 1})
    assert scenario.tests[0].iterations == 3
    assert second.iterations == 5
    assert definition.iterations == 3
    assert definition.extra_env_vars == {"MODE": ["a", "b"]}
    assert first.num_nodes_explicit


def test_snapshot_includes_configuration_and_preserves_subclass(definition: NCCLTestDefinition, tmp_path: Path):
    run = TestRun(test=definition.model_copy(deep=True, update={"id": "case"}), output_path=tmp_path).apply_params_set(
        {"ngpus": 2, "NUM_NODES": 2, "extra_env_vars.MODE": "b"}
    )
    run.current_iteration = 2
    run.step = 4
    payload = toml.loads(toml.dumps(run.model_dump(mode="json")))
    assert set(payload) == {
        "nodes",
        "output_path",
        "current_iteration",
        "step",
        "test",
    }
    assert payload["test"]["id"] == "case"
    assert payload["test"]["iterations"] == 3
    assert payload["test"]["num_nodes"] == 2
    loaded = TestRun.model_validate(payload)
    assert isinstance(loaded.test, NCCLTestDefinition)
    assert loaded.test.cmd_args.ngpus == 2
    assert loaded.nnodes == 2
    assert loaded.current_iteration == 2
    assert loaded.step == 4


def test_scenario_dump_preserves_workload_subclass_fields():
    class CustomDefinition(TestDefinition):
        custom: str = Field(default="custom value")

    definition = CustomDefinition(
        id="case", name="custom", description="custom", test_template_name="custom", cmd_args=CmdArgs()
    )
    scenario = TestScenario(name="scenario", tests=[definition])
    assert scenario.model_dump(by_alias=True)["Tests"][0]["custom"] == "custom value"


@pytest.mark.parametrize(
    "dependencies", [[{"type": "end_post_comp", "id": "case"}], [{"type": "end_post_comp", "id": "unknown"}]]
)
def test_canonical_scenario_validates_dependency_graph(definition: NCCLTestDefinition, dependencies: list[dict]):
    data = definition.model_dump()
    data["dependencies"] = dependencies
    with pytest.raises(ValueError):
        TestScenario(name="scenario", tests=[NCCLTestDefinition.model_validate(data)])


def test_multiple_runtime_trials_share_one_scenario_case(definition: NCCLTestDefinition):
    first = TestRun(test=definition.model_copy(deep=True, update={"id": None}))
    second = first.apply_params_set({"ngpus": 2, "NUM_NODES": 2, "extra_env_vars.MODE": "b"})
    scenario = TestScenario(name="scenario", test_runs=[first, second])
    assert len(scenario.tests) == 1
    assert len(scenario.test_runs) == 2
    assert scenario.tests[0].id == definition.name
    assert first.test.id is None


@pytest.mark.parametrize("reference", ["test_name", "path"])
@pytest.mark.parametrize(
    ("scenario_args", "expected_args"),
    [
        (None, ["--exclusive"]),
        ("--overcommit", ["--overcommit"]),
        (["--overcommit"], ["--overcommit"]),
        ("", []),
        ([], []),
    ],
)
def test_scenario_srun_args_override_test_defaults_for_workload_and_metadata(
    slurm_system: SlurmSystem,
    tmp_path: Path,
    reference: str,
    scenario_args: str | list[str] | None,
    expected_args: list[str],
):
    definition = SlurmContainerTestDefinition(
        name="container",
        description="container",
        test_template_name="SlurmContainer",
        cmd_args=SlurmContainerCmdArgs(docker_image_url="fake://container", cmd="true"),
        extra_srun_args=["--exclusive"],
    )
    slurm_system.extra_srun_args = "--comment=system-flags"
    (tmp_path / "container.toml").write_text(toml.dumps(definition.model_dump()))
    parser = TestScenarioParser(tmp_path / "scenario.toml", slurm_system, {"container": definition}, {})
    case: dict[str, str | list[str]] = {
        "id": "case",
        reference: "container" if reference == "test_name" else "container.toml",
    }
    if scenario_args is not None:
        case["extra_srun_args"] = scenario_args
    scenario = parser._parse_data({"name": "scenario", "Tests": [case]})
    run = scenario.test_runs[0]
    assert run.extra_srun_args == (" ".join(expected_args) or None)
    assert run.test.extra_srun_args == expected_args
    assert "srun_args" not in run.test.model_dump()
    assert definition.extra_srun_args == ["--exclusive"]
    run.output_path = tmp_path
    strategy = SlurmContainerCommandGenStrategy(slurm_system, run)
    commands = [" ".join(strategy.gen_srun_prefix()), *strategy._metadata_cmd().splitlines()]
    for command in commands:
        assert command.count(slurm_system.extra_srun_args) == 1
        for flag in ["--exclusive", "--overcommit"]:
            assert command.count(flag) == (1 if flag in expected_args else 0)
        for flag in expected_args:
            assert command.index(slurm_system.extra_srun_args) < command.index(flag)


@pytest.mark.parametrize("value", ["--comment='a b' --exclusive", ["--comment='a b'", "--exclusive"]])
def test_srun_args_normalization_preserves_shell_quoting(definition: NCCLTestDefinition, value: str | list[str]):
    prepared = type(definition).model_validate(definition.model_dump() | {"extra_srun_args": value})
    assert isinstance(prepared.extra_srun_args, list)
    assert TestRun(test=prepared).extra_srun_args == "--comment='a b' --exclusive"


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("sweep", [False, True])
def test_report_restore_preserves_execution_configuration(
    definition: NCCLTestDefinition,
    slurm_system: SlurmSystem,
    tmp_path: Path,
    legacy: bool,
    sweep: bool,
):
    definition.extra_srun_args = ["--exclusive"]
    definition.exclude_nodes = ["excluded"]
    definition.ideal_perf = 2.0
    template = TestRun(test=definition.model_copy(deep=True, update={"id": "case"}))
    if not sweep:
        template = template.apply_params_set({"ngpus": 2, "NUM_NODES": 2, "extra_env_vars.MODE": "b"})
    selected = template.apply_params_set({"ngpus": 2, "NUM_NODES": 2, "extra_env_vars.MODE": "b"})
    selected.output_path = tmp_path / "case" / "0"
    if sweep:
        selected.output_path /= "1"
        selected.step = 1
    selected.output_path.mkdir(parents=True)
    selected.test = selected.test.model_copy(update={"iterations": 4})
    payload = selected.model_dump(mode="json")
    if legacy:
        payload = {
            "name": selected.name,
            "nnodes": selected.nnodes,
            "iterations": selected.iterations,
            "nodes": selected.nodes,
            "output_path": str(selected.output_path),
            "current_iteration": selected.current_iteration,
            "step": selected.step,
            "test_cmd": "old command",
            "full_cmd": "old launch command",
            "test_definition": selected.test.model_dump(
                include={"name", "description", "test_template_name", "cmd_args", "extra_env_vars"}
            ),
        }
    (selected.output_path / "test-run.toml").write_text(toml.dumps(payload))
    scenario = TestScenario(name="scenario", tests=[definition], test_runs=[template])
    reporter = DSEReporter(slurm_system, scenario, tmp_path, ReportConfig())
    reporter.load_test_runs()
    restored = reporter.trs[0]
    assert restored.iterations == 4
    assert restored.extra_srun_args == "--exclusive"
    assert restored.exclude_nodes == ["excluded"]
    assert restored.test.ideal_perf == 2.0
    assert restored.nnodes == 2
    assert restored.test.cmd_args.ngpus == 2
    assert restored.test.extra_env_vars["MODE"] == "b"
    assert definition.num_nodes == [1, 2]
