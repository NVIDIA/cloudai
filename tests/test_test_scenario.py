# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Set, Type
from unittest.mock import Mock, patch

import pytest
import toml

from cloudai import (
    CmdArgs,
    GitRepo,
    PredictorConfig,
    Registry,
    Test,
    TestDefinition,
    TestRun,
    TestScenario,
    TestScenarioParser,
    TestScenarioParsingError,
)
from cloudai._core.report_generation_strategy import ReportGenerationStrategy
from cloudai._core.test_scenario_parser import calculate_total_time_limit, get_reporters
from cloudai._core.test_template import TestTemplate
from cloudai.models.scenario import TestRunModel, TestScenarioModel
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.chakra_replay import ChakraReplayReportGenerationStrategy, ChakraReplayTestDefinition
from cloudai.workloads.jax_toolbox import (
    GPTTestDefinition,
    GrokTestDefinition,
    JaxToolboxReportGenerationStrategy,
    NemotronTestDefinition,
)
from cloudai.workloads.megatron_run import (
    CheckpointTimingReportGenerationStrategy,
    MegatronRunCmdArgs,
    MegatronRunTestDefinition,
)
from cloudai.workloads.nccl_test import (
    NCCLCmdArgs,
    NCCLTestDefinition,
    NcclTestPerformanceReportGenerationStrategy,
    NcclTestPredictionReportGenerationStrategy,
)
from cloudai.workloads.nemo_launcher import NeMoLauncherReportGenerationStrategy, NeMoLauncherTestDefinition
from cloudai.workloads.nemo_run import (
    NeMoRunDataStoreReportGenerationStrategy,
    NeMoRunReportGenerationStrategy,
    NeMoRunTestDefinition,
)
from cloudai.workloads.sleep import SleepReportGenerationStrategy, SleepTestDefinition
from cloudai.workloads.slurm_container import SlurmContainerReportGenerationStrategy, SlurmContainerTestDefinition
from cloudai.workloads.triton_inference import TritonInferenceReportGenerationStrategy, TritonInferenceTestDefinition
from cloudai.workloads.ucc_test import UCCTestDefinition, UCCTestReportGenerationStrategy


@pytest.fixture
def test_scenario_parser(slurm_system: SlurmSystem) -> TestScenarioParser:
    tsp = TestScenarioParser(Path(""), slurm_system, {}, {})
    return tsp


@pytest.fixture
def test(slurm_system: SlurmSystem) -> Test:
    return Test(
        test_definition=NCCLTestDefinition(
            name="t1",
            description="desc1",
            test_template_name="NcclTest",
            cmd_args=NCCLCmdArgs(),
        ),
        test_template=TestTemplate(system=slurm_system),
    )


def test_single_test_case(test: Test, test_scenario_parser: TestScenarioParser) -> None:
    test_scenario_parser.test_mapping = {"nccl": test}
    test_scenario = test_scenario_parser._parse_data({"name": "nccl-test", "Tests": [{"id": "1", "test_name": "nccl"}]})
    assert test_scenario.name == "nccl-test"
    assert len(test_scenario.test_runs) == 1
    assert test_scenario.job_status_check is True

    tr = test_scenario.test_runs[0]
    assert tr.name == "1"
    assert tr.iterations == 1
    assert tr.current_iteration == 0
    assert tr.dependencies == {}
    assert tr.weight == 0
    assert tr.ideal_perf == 1.0
    assert tr.sol is None
    atest = test_scenario.test_runs[0].test
    assert atest.name == test.name
    assert atest.description == test.description
    assert atest.test_template == test.test_template
    assert atest.cmd_args == test.cmd_args
    assert atest.extra_env_vars == test.extra_env_vars
    assert atest.extra_cmd_args == test.extra_cmd_args


@pytest.mark.parametrize(
    "prop,tvalue,cfg_value",
    [("sol", 1.0, 42.0), ("ideal_perf", 1.0, 42.0)],
)
def test_with_some_props(
    prop: str, tvalue: float, cfg_value: float, test: Test, test_scenario_parser: TestScenarioParser
) -> None:
    setattr(test, prop, tvalue)
    test_scenario_parser.test_mapping = {"nccl": test}
    test_scenario = test_scenario_parser._parse_data(
        {"name": "nccl-test", "Tests": [{"id": "1", "test_name": "nccl", prop: cfg_value}]}
    )
    atest = test_scenario.test_runs[0]
    val = getattr(atest, prop)
    assert val != tvalue
    assert val == cfg_value


def test_with_time_limit(test: Test, test_scenario_parser: TestScenarioParser) -> None:
    test_scenario_parser.test_mapping = {"nccl": test}
    test_scenario = test_scenario_parser._parse_data(
        {"name": "nccl-test", "Tests": [{"id": "1", "test_name": "nccl", "time_limit": "10m"}]}
    )
    assert test_scenario.test_runs[0].time_limit == "00:10:00"


def test_two_independent_cases(test: Test, test_scenario_parser: TestScenarioParser) -> None:
    t1, t2 = test, test

    test_scenario_parser.test_mapping = {"nccl": t1, "nccl2": t2}
    test_scenario = test_scenario_parser._parse_data(
        {"name": "nccl-test", "Tests": [{"id": "1", "test_name": "nccl"}, {"id": "2", "test_name": "nccl2"}]}
    )
    assert len(test_scenario.test_runs) == 2

    assert test_scenario.test_runs[0].test.name == t1.name
    assert test_scenario.test_runs[0].dependencies == {}

    assert test_scenario.test_runs[1].test.name == t2.name
    assert test_scenario.test_runs[1].dependencies == {}


def test_cant_depends_on_itself() -> None:
    with pytest.raises(ValueError) as exc_info:
        TestScenarioModel.model_validate(
            {
                "name": "nccl-test",
                "Tests": [
                    {
                        "id": "1",
                        "test_name": "nccl",
                        "dependencies": [{"type": "end_post_comp", "id": "1"}],
                    },
                ],
            }
        )
    assert exc_info.match("Test '1' must not depend on itself.")


def test_two_dependent_cases(test: Test, test_scenario_parser: TestScenarioParser) -> None:
    t1, t2 = test, test

    test_scenario_parser.test_mapping = {"nccl": t1, "nccl2": t2}
    test_scenario = test_scenario_parser._parse_data(
        {
            "name": "nccl-test",
            "Tests": [
                {"id": "1", "test_name": "nccl", "dependencies": [{"type": "end_post_comp", "id": "2"}]},
                {"id": "2", "test_name": "nccl2"},
            ],
        }
    )
    assert len(test_scenario.test_runs) == 2

    assert test_scenario.test_runs[0].test.name == t1.name
    assert "end_post_comp" in test_scenario.test_runs[0].dependencies
    assert isinstance(test_scenario.test_runs[0].dependencies["end_post_comp"].test_run, TestRun)

    assert test_scenario.test_runs[1].test.name == t2.name
    assert test_scenario.test_runs[1].dependencies == {}


def test_ids_must_be_unique() -> None:
    with pytest.raises(ValueError) as exc_info:
        TestScenarioModel.model_validate(
            {
                "name": "test",
                "Tests": [
                    {"id": "1", "test_name": "nccl"},
                    {"id": "1", "test_name": "nccl"},
                ],
            }
        )
    assert exc_info.match("Duplicate test id '1' found in the test scenario.")


def test_raises_on_unknown_dependency() -> None:
    with pytest.raises(ValueError) as exc_info:
        TestScenarioModel.model_validate(
            {
                "name": "test",
                "Tests": [
                    {
                        "id": "test-id",
                        "test_name": "nccl",
                        "dependencies": [{"type": "end_post_comp", "id": "dep-id"}],
                    }
                ],
            }
        )

    assert exc_info.match("Dependency section 'dep-id' not found for test 'test-id'.")


def test_list_of_tests_must_not_be_empty() -> None:
    with pytest.raises(ValueError) as exc_info:
        TestScenarioModel.model_validate({"name": "name"})
    assert exc_info.match("TestScenarioModel\nTests\n  Field required")

    with pytest.raises(ValueError) as exc_info:
        TestScenarioModel.model_validate({"name": "name", "Tests": []})
    assert exc_info.match("TestScenarioModel\nTests\n  List should have at least 1 item after validation")


def test_test_id_must_contain_at_least_one_letter() -> None:
    with pytest.raises(ValueError) as exc_info:
        TestScenarioModel.model_validate({"name": "name", "Tests": [{"id": "", "test_name": "nccl"}]})
    assert exc_info.match("TestScenarioModel\nTests.0.id\n  String should have at least 1 character")


@pytest.mark.parametrize(
    "time_str, expected",
    [
        ("10m", "00:10:00"),
        ("1h", "01:00:00"),
        ("2d", "2-00:00:00"),
        ("1w", "7-00:00:00"),
        ("30s", "00:00:30"),
        ("1-12:30:45", "1-12:30:45"),
        ("12:30:45", "12:30:45"),
        ("12:30", "12:30:00"),
    ],
)
def test_calculate_total_time_limit(time_str, expected):
    assert calculate_total_time_limit([], time_limit=time_str) == expected


def test_create_test_run_with_hooks(test: Test, test_scenario_parser: TestScenarioParser):
    pre_test = TestScenario(
        name="pre",
        test_runs=[TestRun(name="pre1", test=test, num_nodes=1, nodes=[], time_limit="00:30:00", iterations=1)],
    )
    post_test = TestScenario(
        name="post",
        test_runs=[TestRun(name="post1", test=test, num_nodes=1, nodes=[], time_limit="00:20:00", iterations=1)],
    )

    test_info = TestRunModel(id="main1", test_name="test1", time_limit="01:00:00", weight=10, iterations=1, num_nodes=1)
    test_scenario_parser.test_mapping = {"test1": test}

    test_run = test_scenario_parser._create_test_run(
        test_info=test_info, normalized_weight=1.0, pre_test=pre_test, post_test=post_test
    )

    assert test_run.time_limit == "01:50:00"  # Main + pre + post hooks


def test_total_time_limit_with_empty_hooks():
    result = calculate_total_time_limit([], "01:00:00")
    assert result == "01:00:00"


class TestInScenario:
    @pytest.mark.parametrize("missing_arg", ["test_template_name", "name", "description"])
    def test_without_base(self, missing_arg: str):
        spec = {
            "id": "1",
            "test_template_name": "NcclTest",
            "name": "nccl",
            "description": "desc",
        }
        spec.pop(missing_arg)
        with pytest.raises(ValueError) as exc_info:
            TestRunModel.model_validate(spec)
        assert exc_info.match(
            "When 'test_name' is not set, the following fields must be set: 'test_template_name', 'name', 'description'"
        )

    def test_name_is_not_in_mapping(self, test_scenario_parser: TestScenarioParser):
        with pytest.raises(ValueError) as exc_info:
            test_scenario_parser._prepare_tdef(TestRunModel(id="1", test_name="nccl"))
        assert exc_info.match("Test 'nccl' is not defined. Was tests directory correctly set?")

    @pytest.mark.parametrize("override_arg", ["name", "description"])
    def test_can_override_name_and_description(self, override_arg: str):
        spec = {"id": "1", "test_name": "nccl", override_arg: "value"}
        model = TestRunModel.model_validate(spec)

        data = model.model_dump()
        assert data[override_arg] == "value"

    def test_cant_override_template_name(self):
        spec = {"id": "1", "test_name": "nccl", "test_template_name": "NcclTest"}
        with pytest.raises(ValueError) as exc_info:
            TestRunModel.model_validate(spec)
        assert exc_info.match("'test_template_name' must not be set if 'test_name' is set.")

    def test_spec_with_unknown_test_type(self):
        with pytest.raises(ValueError) as exc_info:
            TestRunModel(id="1", name="nccl", description="desc", test_template_name="unknown")
        assert exc_info.match("Test type 'unknown' not found in the test definitions. Possible values are:")

    def test_type_is_not_allowed_when_name_is_set(self):
        with pytest.raises(ValueError) as exc_info:
            TestRunModel(id="1", test_name="nccl", test_template_name="NcclTest")
        assert exc_info.match("'test_template_name' must not be set if 'test_name' is set.")

    def test_spec_without_base(self, test_scenario_parser: TestScenarioParser):
        model = TestScenarioModel.model_validate(
            toml.loads(
                """
            name = "test"

            [[Tests]]
            id = "1"
            name = "nccl"
            description = "desc"
            test_template_name = "NcclTest"
            """
            )
        )
        assert model.name == "test"
        assert len(model.tests) == 1
        assert model.tests[0].name == "nccl"
        assert model.tests[0].test_template_name == "NcclTest"
        assert model.tests[0].description == "desc"

    def test_spec_has_priority(self, test_scenario_parser: TestScenarioParser, slurm_system: SlurmSystem):
        test_scenario_parser.test_mapping = {
            "nccl": Test(
                test_definition=NCCLTestDefinition(
                    name="nccl", description="desc", test_template_name="NcclTest", cmd_args=NCCLCmdArgs()
                ),
                test_template=TestTemplate(system=slurm_system),
            )
        }
        model = TestScenarioModel.model_validate(
            toml.loads(
                """
            name = "test"

            [[Tests]]
            id = "1"
            test_name = "nccl"

              [Tests.cmd_args]
              nthreads = 42
            """
            )
        )
        _, tdef = test_scenario_parser._prepare_tdef(model.tests[0])
        assert tdef.cmd_args.nthreads == 42

    def test_spec_can_set_unknown_args(self, test_scenario_parser: TestScenarioParser, slurm_system: SlurmSystem):
        test_scenario_parser.test_mapping = {
            "nccl": Test(
                test_definition=NCCLTestDefinition(
                    name="nccl", description="desc", test_template_name="NcclTest", cmd_args=NCCLCmdArgs()
                ),
                test_template=TestTemplate(system=slurm_system),
            )
        }
        model = TestScenarioModel.model_validate(
            toml.loads(
                """
            name = "test"

            [[Tests]]
            id = "1"
            test_name = "nccl"
            cmd_args = { unknown = 42 }
            """
            )
        )
        _, tdef = test_scenario_parser._prepare_tdef(model.tests[0])
        assert tdef.cmd_args_dict["unknown"] == 42

    def test_spec_can_set_unknown_args_no_base(self, test_scenario_parser: TestScenarioParser):
        model = TestScenarioModel.model_validate(
            toml.loads(
                """
            name = "test"

            [[Tests]]
            id = "1"
            name = "nccl"
            test_template_name = "NcclTest"
            description = "desc"
            cmd_args = { unknown = 42 }
            """
            )
        )
        _, tdef = test_scenario_parser._prepare_tdef(model.tests[0])
        assert tdef.cmd_args_dict["unknown"] == 42
        assert isinstance(tdef.cmd_args, NCCLCmdArgs)

    def test_data_is_merge_correctly(self, test_scenario_parser: TestScenarioParser, slurm_system: SlurmSystem):
        test_scenario_parser.test_mapping = {
            "megatron": Test(
                test_definition=MegatronRunTestDefinition(
                    name="megatron",
                    description="desc",
                    test_template_name="MegatronRun",
                    cmd_args=MegatronRunCmdArgs(docker_image_url="docker://megatron", run_script=Path("run.sh")),
                ),
                test_template=TestTemplate(system=slurm_system),
            )
        }
        model = TestScenarioModel.model_validate(
            toml.loads(
                """
            name = "test"

            [[Tests]]
            id = "1"
            test_name = "megatron"
            cmd_args = { any = 42 }
            """
            )
        )
        _, tdef = test_scenario_parser._prepare_tdef(model.tests[0])
        assert isinstance(tdef.cmd_args, MegatronRunCmdArgs)
        assert tdef.cmd_args.run_script == Path("run.sh")


class TestReporters:
    def test_default(self):
        reporters = get_reporters(
            TestRunModel(id="id", test_name="tn"),
            TestDefinition(name="test", description="desc", test_template_name="tt", cmd_args=CmdArgs()),
        )
        assert len(reporters) == 0

    def test_default_reporters_size(self):
        assert len(Registry().reports_map) == 12

    @pytest.mark.parametrize(
        "tdef,expected_reporters",
        [
            (ChakraReplayTestDefinition, {ChakraReplayReportGenerationStrategy}),
            (GPTTestDefinition, {JaxToolboxReportGenerationStrategy}),
            (GrokTestDefinition, {JaxToolboxReportGenerationStrategy}),
            (MegatronRunTestDefinition, {CheckpointTimingReportGenerationStrategy}),
            (NCCLTestDefinition, {NcclTestPerformanceReportGenerationStrategy}),
            (NeMoLauncherTestDefinition, {NeMoLauncherReportGenerationStrategy}),
            (NeMoRunTestDefinition, {NeMoRunReportGenerationStrategy, NeMoRunDataStoreReportGenerationStrategy}),
            (NemotronTestDefinition, {JaxToolboxReportGenerationStrategy}),
            (SleepTestDefinition, {SleepReportGenerationStrategy}),
            (SlurmContainerTestDefinition, {SlurmContainerReportGenerationStrategy}),
            (UCCTestDefinition, {UCCTestReportGenerationStrategy}),
            (TritonInferenceTestDefinition, {TritonInferenceReportGenerationStrategy}),
        ],
    )
    def test_custom_reporters(self, tdef: Type[TestDefinition], expected_reporters: Set[ReportGenerationStrategy]):
        assert Registry().reports_map[tdef] == expected_reporters

    def test_get_reporters_nccl(self):
        tr_model = TestRunModel(id="id", test_name="nccl", time_limit="01:00:00", weight=10, iterations=1, num_nodes=1)
        tdef = NCCLTestDefinition(name="nccl", description="desc", test_template_name="tt", cmd_args=NCCLCmdArgs())
        reporters = get_reporters(tr_model, tdef)
        assert len(reporters) == 1
        assert NcclTestPerformanceReportGenerationStrategy in reporters

        tdef.predictor = PredictorConfig(git_repo=GitRepo(url="", commit=""))
        reporters = get_reporters(tr_model, tdef)
        assert len(reporters) == 2
        assert NcclTestPerformanceReportGenerationStrategy in reporters
        assert NcclTestPredictionReportGenerationStrategy in reporters


class TestReportMetricsDSE:
    @pytest.fixture
    def tname(self) -> str:
        return "nccl"

    @pytest.fixture
    def test_info(self, tname: str) -> TestRunModel:
        return TestRunModel(id="main1", test_name=tname, time_limit="01:00:00", weight=10, iterations=1, num_nodes=1)

    @pytest.fixture
    def ts_parser(self, test_scenario_parser: TestScenarioParser) -> TestScenarioParser:
        nccl = NCCLTestDefinition(
            name="nccl",
            description="desc",
            test_template_name="NcclTest",
            cmd_args=NCCLCmdArgs(),
            extra_env_vars={"DSE": ["v1", "v2"]},
        )
        test_scenario_parser.test_mapping["nccl"] = Test(test_definition=nccl, test_template=Mock())
        return test_scenario_parser

    def test_raises_on_unknown_metric(
        self, ts_parser: TestScenarioParser, tname: str, test_info: TestRunModel, caplog: pytest.LogCaptureFixture
    ):
        tdef = ts_parser.test_mapping[tname].test_definition
        tdef.agent_metric = "unknown"

        with pytest.raises(TestScenarioParsingError) as exc_info:
            ts_parser._create_test_run(test_info=test_info, normalized_weight=1.0)

        mapping_str = (
            f"{NcclTestPerformanceReportGenerationStrategy}: {NcclTestPerformanceReportGenerationStrategy.metrics}"
        )
        msg = (
            f"Test '{test_info.id}' is a DSE job with agent_metric='{tdef.agent_metric}', "
            "but no report generation strategy is defined for it. "
            f"Available report-metrics mapping: {{{mapping_str}}}"
        )
        assert str(exc_info.value) == msg
        assert caplog.records[0].levelname == "ERROR"
        assert caplog.records[0].message == f"Failed to parse Test Scenario definition: {ts_parser.file_path}"
        assert caplog.records[1].levelname == "ERROR"
        assert caplog.records[1].message == msg

    @patch("cloudai._core.test_scenario_parser.get_reporters", return_value=set())
    def test_raises_if_no_reports_defined(self, _, ts_parser: TestScenarioParser, test_info: TestRunModel, tname: str):
        tdef = ts_parser.test_mapping[tname].test_definition
        tdef.agent_metric = "default"

        with pytest.raises(TestScenarioParsingError) as exc_info:
            ts_parser._create_test_run(test_info=test_info, normalized_weight=1.0)

        assert str(exc_info.value) == (
            f"Test '{test_info.id}' is a DSE job with agent_metric='{tdef.agent_metric}', "
            "but no report generation strategy is defined for it. "
            "Available report-metrics mapping: {}"
        )
