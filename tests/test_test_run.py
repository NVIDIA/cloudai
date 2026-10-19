# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import ClassVar

import pytest
import toml
from pydantic import ValidationError

from cloudai.core import ReportGenerationStrategy, TestDefinition, TestDependency, TestRun, TestScenario, load_test_run
from cloudai.models.workload import CmdArgs, TestRunDependencyModel
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition


class MyReport(ReportGenerationStrategy):
    metrics: ClassVar[list[str]] = ["default", "metric1"]

    def can_handle_directory(self) -> bool:
        return True

    def generate_report(self) -> None:
        return


class MyReport2(ReportGenerationStrategy):
    metrics: ClassVar[list[str]] = ["default", "metric2"]

    def can_handle_directory(self) -> bool:
        return True

    def generate_report(self) -> None:
        return


class TestMetricsReporter:
    @pytest.fixture
    def tr(self) -> TestRun:
        return TestRun(
            test=(
                TestDefinition(name="test", description="", test_template_name="Test", cmd_args=CmdArgs())
            ).model_copy(deep=True, update={"id": "test", "num_nodes": 1}),
            nodes=[],
        )

    def test_no_reports(self, tr: TestRun):
        tr.reports.clear()
        tr.test.agent_metrics = ["default"]
        assert tr.metric_reporter is None

    def test_no_metrics(self, tr: TestRun):
        tr.test = tr.test.model_copy(update={"reporters": {MyReport}})
        tr.test.agent_metrics = []
        assert tr.metric_reporter is None

    def test_one_report_multiple_metrics(self, tr: TestRun):
        tr.test = tr.test.model_copy(update={"reporters": {MyReport}})
        metrics = ["m1", "m2"]
        MyReport.metrics = metrics
        tr.test.agent_metrics = metrics

        assert tr.metric_reporter is MyReport

    def test_all_metrics_should_be_provided_by_the_same_reporter(self, tr: TestRun):
        tr.test = tr.test.model_copy(update={"reporters": {MyReport, MyReport2}})
        metrics = ["m1", "m2"]
        MyReport.metrics = ["m1"]
        MyReport2.metrics = ["m2"]
        tr.test.agent_metrics = metrics

        assert tr.metric_reporter is None


@pytest.fixture
def run(tmp_path: Path) -> TestRun:
    definition = NCCLTestDefinition(
        id="case",
        name="nccl",
        description="test",
        test_template_name="NcclTest",
        num_nodes=2,
        iterations=4,
        time_limit="00:30:00",
        weight=5,
        cmd_args=NCCLCmdArgs(docker_image_url="fake://nccl", ngpus=2),
    )
    return TestRun(test=definition, nodes=["node1", "node2"], output_path=tmp_path, current_iteration=1, step=3)


def test_direct_dump_excludes_live_bindings_and_restores_workload(run: TestRun) -> None:
    run.test.dependencies = [TestRunDependencyModel(type="end_post_comp", id="other")]
    run.dependencies = {"end_post_comp": TestDependency(run)}
    run.pre_test = TestScenario(name="pre", tests=[])
    run.post_test = TestScenario(name="post", tests=[])
    payload = run.model_dump(mode="json")
    assert set(payload) == {"test", "nodes", "output_path", "current_iteration", "step"}
    assert payload["test"]["dependencies"] == [{"type": "end_post_comp", "id": "other"}]
    path = run.output_path / "test-run.toml"
    path.write_text(toml.dumps(payload))

    loaded = load_test_run(path)

    assert isinstance(loaded.test, NCCLTestDefinition)
    assert loaded.test == run.test
    assert loaded.nodes == run.nodes
    assert loaded.current_iteration == 1
    assert loaded.step == 3
    assert loaded.pre_test is loaded.post_test is None
    assert loaded.dependencies == {}


@pytest.mark.parametrize("legacy", [False, True])
def test_loader_supports_old_and_new_files_in_moved_folders(run: TestRun, tmp_path: Path, legacy: bool) -> None:
    fallback = run.test.model_copy(update={"iterations": 99, "num_nodes": 8, "weight": 99, "reporters": {object}})
    payload = run.model_dump(mode="json")
    if legacy:
        payload = {
            "name": run.name,
            "nnodes": run.nnodes,
            "iterations": run.iterations,
            "nodes": run.nodes,
            "output_path": "/old/results",
            "current_iteration": run.current_iteration,
            "step": run.step,
            "test_cmd": "ignored",
            "full_cmd": "ignored",
            "test_definition": run.test.model_dump(include={"name", "description", "test_template_name", "cmd_args"}),
        }
    moved = tmp_path / "moved"
    moved.mkdir()
    path = moved / "test-run.toml"
    path.write_text(toml.dumps(payload))

    loaded = load_test_run(path, fallback)

    assert loaded.name == "case"
    assert loaded.nnodes == 2
    assert loaded.iterations == 4
    assert loaded.test.weight == (99 if legacy else 5)
    assert loaded.test.reporters == {object}
    assert loaded.nodes == ["node1", "node2"]
    assert loaded.output_path == moved
    assert loaded.current_iteration == 1
    assert loaded.step == 3
    assert fallback.num_nodes == 8
    assert fallback.iterations == 99


def test_new_files_do_not_fill_missing_fields_from_scenario(run: TestRun) -> None:
    payload = run.model_dump(mode="json")
    del payload["test"]["name"]
    path = run.output_path / "test-run.toml"
    path.write_text(toml.dumps(payload))
    with pytest.raises(ValidationError, match="name"):
        load_test_run(path, run.test)


def test_materialized_environment_parameters_round_trip(run: TestRun) -> None:
    data = run.test.model_dump()
    data["cmd_args"]["ngpus"] = [1, 2]
    data["env_params"] = {"ngpus": {"weights": [0.5, 0.5]}}
    template = TestRun(test=NCCLTestDefinition.model_validate(data), output_path=run.output_path)
    selected = template.apply_params_set({}, env_params={"ngpus": 2})
    path = run.output_path / "test-run.toml"
    path.write_text(toml.dumps(selected.model_dump(mode="json")))

    loaded = load_test_run(path)

    assert loaded.test.cmd_args.ngpus == 2
    assert loaded.test.env_params["ngpus"].weights == [0.5, 0.5]
    assert template.test.cmd_args.ngpus == [1, 2]


@pytest.mark.parametrize("nodes", [None, [], ["allocated-node"]])
def test_runtime_nodes_default_to_definition_without_sharing_list(run: TestRun, nodes: list[str] | None) -> None:
    definition = run.test.model_copy(update={"nodes": ["configured-node"]})
    loaded = TestRun(test=definition) if nodes is None else TestRun(test=definition, nodes=nodes)
    assert loaded.nodes == (["configured-node"] if nodes is None else nodes)
    loaded.nodes.append("another-node")
    assert definition.nodes == ["configured-node"]
