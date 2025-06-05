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

import copy

import pytest

from cloudai.configurator import BaseAgent
from cloudai.core import (
    BaseInstaller,
    BaseRunner,
    JobIdRetrievalStrategy,
    JobStatusRetrievalStrategy,
    Registry,
    Reporter,
    ReportGenerationStrategy,
    System,
    TestTemplateStrategy,
)
from cloudai.models.workload import TestDefinition


class MyTestDefinition(TestDefinition):
    pass


@pytest.fixture
def registry():
    registry = Registry()
    scenario_reports = copy.copy(registry.scenario_reports)
    registry.scenario_reports.clear()

    yield registry

    # Clean up the registry after the test, we check exact list of reports in other tests
    if MyTestDefinition in registry.reports_map:
        del registry.reports_map[MyTestDefinition]
    registry.update_scenario_report(scenario_reports)


class MyRunner(BaseRunner):
    pass


class AnotherRunner(BaseRunner):
    pass


class TestRegistry__RunnersMap:
    """This test verifies Registry class functionality.

    Since Registry is a Singleton, the order of cases is important.
    Only covers the runners_map attribute.
    """

    def test_add_runner(self, registry: Registry):
        registry.add_runner("runner", MyRunner)
        assert registry.runners_map["runner"] == MyRunner

    def test_add_runner_duplicate(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.add_runner("runner", MyRunner)
        assert "Duplicating implementation for 'runner'" in str(exc_info.value)

    def test_update_runner(self, registry: Registry):
        registry.update_runner("runner", AnotherRunner)
        assert registry.runners_map["runner"] == AnotherRunner


class MyStrategy(TestTemplateStrategy):
    pass


class MySystem(System):
    pass


class AnotherSystem(System):
    pass


class AnotherStrategy(TestTemplateStrategy):
    pass


class MyJobIdRetrievalStrategy(JobIdRetrievalStrategy):
    pass


class MyJobStatusRetrievalStrategy(JobStatusRetrievalStrategy):
    pass


class TestRegistry__StrategiesMap:
    """This test verifies Registry class functionality.

    Since Registry is a Singleton, the order of cases is important.
    Only covers the strategies_map attribute.
    """

    def test_add_strategy(self, registry: Registry):
        registry.add_strategy(MyStrategy, [MySystem], [MyTestDefinition], MyStrategy)
        registry.add_strategy(MyJobIdRetrievalStrategy, [MySystem], [MyTestDefinition], MyJobIdRetrievalStrategy)
        registry.add_strategy(
            MyJobStatusRetrievalStrategy, [MySystem], [MyTestDefinition], MyJobStatusRetrievalStrategy
        )

        assert registry.strategies_map[(MyStrategy, MySystem, MyTestDefinition)] == MyStrategy
        assert (
            registry.strategies_map[(MyJobIdRetrievalStrategy, MySystem, MyTestDefinition)] == MyJobIdRetrievalStrategy
        )
        assert (
            registry.strategies_map[(MyJobStatusRetrievalStrategy, MySystem, MyTestDefinition)]
            == MyJobStatusRetrievalStrategy
        )

    def test_add_strategy_duplicate(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.add_strategy(MyStrategy, [MySystem], [MyTestDefinition], MyStrategy)
        assert "Duplicating implementation for" in str(exc_info.value)

    def test_update_strategy(self, registry: Registry):
        registry.update_strategy((MyStrategy, MySystem, MyTestDefinition), AnotherStrategy)
        assert registry.strategies_map[(MyStrategy, MySystem, MyTestDefinition)] == AnotherStrategy

    def test_add_multiple_strategies(self, registry: Registry):
        registry.strategies_map.clear()

        registry.add_strategy(
            MyStrategy, [MySystem, AnotherSystem], [MyTestDefinition, AnotherTestDefinition], MyStrategy
        )
        assert len(registry.strategies_map) == 4
        assert registry.strategies_map[(MyStrategy, MySystem, MyTestDefinition)] == MyStrategy
        assert registry.strategies_map[(MyStrategy, MySystem, AnotherTestDefinition)] == MyStrategy
        assert registry.strategies_map[(MyStrategy, AnotherSystem, MyTestDefinition)] == MyStrategy
        assert registry.strategies_map[(MyStrategy, AnotherSystem, AnotherTestDefinition)] == MyStrategy


class MyInstaller(BaseInstaller):
    pass


class AnotherInstaller(BaseInstaller):
    pass


class TestRegistry__Installers:
    """This test verifies Registry class functionality.

    Since Registry is a Singleton, the order of cases is important.
    Only covers the installers_map attribute.
    """

    def test_add_installer(self, registry: Registry):
        registry.add_installer("installer", MyInstaller)
        assert registry.installers_map["installer"] == MyInstaller

    def test_add_installer_duplicate(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.add_installer("installer", MyInstaller)
        assert "Duplicating implementation for 'installer'" in str(exc_info.value)

    def test_update_installer(self, registry: Registry):
        registry.update_installer("installer", AnotherInstaller)
        assert registry.installers_map["installer"] == AnotherInstaller


class AnotherTestDefinition(TestDefinition):
    pass


class TestRegistry__TestDefinitions:
    """This test verifies Registry class functionality.

    Since Registry is a Singleton, the order of cases is important.
    Only covers the test_definitions_map attribute.
    """

    def test_add_test_definition(self, registry: Registry):
        registry.add_test_definition("test_definition", MyTestDefinition)
        assert registry.test_definitions_map["test_definition"] == MyTestDefinition

    def test_add_test_definition_duplicate(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.add_test_definition("test_definition", MyTestDefinition)
        assert "Duplicating implementation for 'test_definition'" in str(exc_info.value)

    def test_update_test_definition(self, registry: Registry):
        registry.update_test_definition("test_definition", AnotherTestDefinition)
        assert registry.test_definitions_map["test_definition"] == AnotherTestDefinition


class MyAgent(BaseAgent):
    pass


class AnotherAgent(BaseAgent):
    pass


class TestRegistry__AgentsMap:
    """This test verifies Registry class functionality.

    Since Registry is a Singleton, the order of cases is important.
    Only covers the agents_map attribute.
    """

    def test_add_agent(self, registry: Registry):
        registry.add_agent("agent", MyAgent)
        assert registry.agents_map["agent"] == MyAgent

    def test_add_agent_duplicate(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.add_agent("agent", MyAgent)
        assert "Duplicating implementation for 'agent'" in str(exc_info.value)

    def test_update_agent(self, registry: Registry):
        registry.update_agent("agent", AnotherAgent)
        assert registry.agents_map["agent"] == AnotherAgent


class MyReport(ReportGenerationStrategy):
    pass


class AnotherReport(ReportGenerationStrategy):
    pass


class TestRegistry__ReportsMap:
    """This test verifies Registry class functionality.

    Since Registry is a Singleton, the order of cases is important.
    Only covers the reports_map attribute.
    """

    def test_add_report(self, registry: Registry):
        registry.add_report(MyTestDefinition, MyReport)
        assert registry.reports_map[MyTestDefinition] == {MyReport}

    def test_duplicate_is_fine(self, registry: Registry):
        registry.add_report(MyTestDefinition, MyReport)
        registry.add_report(MyTestDefinition, MyReport)
        assert registry.reports_map[MyTestDefinition] == {MyReport}

    def test_add_multiple_reports(self, registry: Registry):
        registry.add_report(MyTestDefinition, MyReport)
        registry.add_report(MyTestDefinition, AnotherReport)
        assert registry.reports_map[MyTestDefinition] == {MyReport, AnotherReport}

    def test_update_report(self, registry: Registry):
        registry.update_report(MyTestDefinition, {AnotherReport})
        assert registry.reports_map[MyTestDefinition] == {AnotherReport}


class MyReporter(Reporter):
    pass


class AnotherReporter(Reporter):
    pass


class TestRegistry__ScenarioReports:
    """This test verifies Registry class functionality.

    Since Registry is a Singleton, the order of cases is important.
    Only covers the scenario_reports attribute.
    """

    def test_add_scenario_report(self, registry: Registry):
        registry.add_scenario_report(MyReporter)
        assert registry.scenario_reports == [MyReporter]

    def test_duplicate_is_fine(self, registry: Registry):
        registry.add_scenario_report(MyReporter)
        registry.add_scenario_report(MyReporter)
        assert registry.scenario_reports == [MyReporter]

    def test_can_add_multiple_reports(self, registry: Registry):
        registry.add_scenario_report(MyReporter)
        registry.add_scenario_report(AnotherReporter)
        assert registry.scenario_reports == [MyReporter, AnotherReporter]

    def test_update_scenario_report(self, registry: Registry):
        registry.update_scenario_report([AnotherReporter])
        assert registry.scenario_reports == [AnotherReporter]
