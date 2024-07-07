#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from cloudai import JobIdRetrievalStrategy, JobStatusRetrievalStrategy, ReportGenerationStrategy, System, TestTemplate
from cloudai._core.base_installer import BaseInstaller
from cloudai._core.base_runner import BaseRunner
from cloudai._core.base_system_parser import BaseSystemParser
from cloudai._core.registry import Registry
from cloudai._core.test_template_strategy import TestTemplateStrategy


class MySystemParser(BaseSystemParser):
    pass


class AnotherSystemParser(BaseSystemParser):
    pass


@pytest.fixture
def registry():
    return Registry()


class TestRegistry__SystemParsersMap:
    """This test verifies Registry class functionality.

    Since Registry is a Singleton, the order of cases is important.
    Only covers the system_parsers_map attribute.
    """

    def test_add_system(self, registry: Registry):
        registry.add_system_parser("system", MySystemParser)
        assert registry.system_parsers_map["system"] == MySystemParser

    def test_add_system_duplicate(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.add_system_parser("system", MySystemParser)
        assert "Duplicating implementation for 'system'" in str(exc_info.value)

    def test_update_system(self, registry: Registry):
        registry.update_system_parser("system", AnotherSystemParser)
        assert registry.system_parsers_map["system"] == AnotherSystemParser

    def test_invalid_type(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.update_system_parser("TestSystem", str)  # pyright: ignore
        assert "Invalid system implementation for 'TestSystem'" in str(exc_info.value)


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

    def test_invalid_type(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.update_runner("TestRunner", str)  # pyright: ignore
        assert "Invalid runner implementation for 'TestRunner'" in str(exc_info.value)


class MyStrategy(TestTemplateStrategy):
    pass


class MySystem(System):
    pass


class AnotherSystem(System):
    pass


class MyTestTemplate(TestTemplate):
    pass


class AnotherTestTemplate(TestTemplate):
    pass


class AnotherStrategy(TestTemplateStrategy):
    pass


class MyReportGenerationStrategy(ReportGenerationStrategy):
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
        registry.add_strategy(MyStrategy, [MySystem], [MyTestTemplate], MyStrategy)
        registry.add_strategy(MyReportGenerationStrategy, [MySystem], [MyTestTemplate], MyReportGenerationStrategy)
        registry.add_strategy(MyJobIdRetrievalStrategy, [MySystem], [MyTestTemplate], MyJobIdRetrievalStrategy)
        registry.add_strategy(MyJobStatusRetrievalStrategy, [MySystem], [MyTestTemplate], MyJobStatusRetrievalStrategy)

        assert registry.strategies_map[(MyStrategy, MySystem, MyTestTemplate)] == MyStrategy
        assert (
            registry.strategies_map[(MyReportGenerationStrategy, MySystem, MyTestTemplate)]
            == MyReportGenerationStrategy
        )
        assert registry.strategies_map[(MyJobIdRetrievalStrategy, MySystem, MyTestTemplate)] == MyJobIdRetrievalStrategy
        assert (
            registry.strategies_map[(MyJobStatusRetrievalStrategy, MySystem, MyTestTemplate)]
            == MyJobStatusRetrievalStrategy
        )

    def test_add_strategy_duplicate(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.add_strategy(MyStrategy, [MySystem], [MyTestTemplate], MyStrategy)
        assert "Duplicating implementation for" in str(exc_info.value)

    def test_update_strategy(self, registry: Registry):
        registry.update_strategy((MyStrategy, MySystem, MyTestTemplate), AnotherStrategy)
        assert registry.strategies_map[(MyStrategy, MySystem, MyTestTemplate)] == AnotherStrategy

    def test_invalid_type__strategy_interface(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.update_strategy((str, MySystem, MyTestTemplate), MyStrategy)  # pyright: ignore
        err = (
            "Invalid strategy interface type, should be subclass of 'TestTemplateStrategy' or "
            "'ReportGenerationStrategy' or 'JobIdRetrievalStrategy' or 'JobStatusRetrievalStrategy'."
        )
        assert err in str(exc_info.value)

    def test_invalid_type__system(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.update_strategy((MyStrategy, str, MyTestTemplate), MyStrategy)  # pyright: ignore
        assert "Invalid system type, should be subclass of 'System'." in str(exc_info.value)

    def test_invalid_type__template(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.update_strategy((MyStrategy, MySystem, str), MyStrategy)  # pyright: ignore
        assert "Invalid test template type, should be subclass of 'TestTemplate'." in str(exc_info.value)

    def test_invalid_type__strategy(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.update_strategy((MyStrategy, MySystem, MyTestTemplate), str)  # pyright: ignore
        assert "Invalid strategy implementation " in str(exc_info.value)
        assert "should be subclass of 'TestTemplateStrategy'." in str(exc_info.value)

    def test_add_multiple_strategies(self, registry: Registry):
        registry.strategies_map = {}

        registry.add_strategy(MyStrategy, [MySystem, AnotherSystem], [MyTestTemplate, AnotherTestTemplate], MyStrategy)
        assert len(registry.strategies_map) == 4
        assert registry.strategies_map[(MyStrategy, MySystem, MyTestTemplate)] == MyStrategy
        assert registry.strategies_map[(MyStrategy, MySystem, AnotherTestTemplate)] == MyStrategy
        assert registry.strategies_map[(MyStrategy, AnotherSystem, MyTestTemplate)] == MyStrategy
        assert registry.strategies_map[(MyStrategy, AnotherSystem, AnotherTestTemplate)] == MyStrategy


class TestRegistry__TestTemplatesMap:
    """This test verifies Registry class functionality.

    Since Registry is a Singleton, the order of cases is important.
    Only covers the test_templates_map attribute.
    """

    def test_add_test_template(self, registry: Registry):
        registry.add_test_template("test_template", MyTestTemplate)
        assert registry.test_templates_map["test_template"] == MyTestTemplate

    def test_add_test_template_duplicate(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.add_test_template("test_template", MyTestTemplate)
        assert "Duplicating implementation for 'test_template'" in str(exc_info.value)

    def test_update_test_template(self, registry: Registry):
        registry.update_test_template("test_template", AnotherTestTemplate)
        assert registry.test_templates_map["test_template"] == AnotherTestTemplate

    def test_invalid_type(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.update_test_template("TestTemplate", str)  # pyright: ignore
        assert "Invalid test template implementation for 'TestTemplate'" in str(exc_info.value)


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

    def test_invalid_type(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.update_installer("TestInstaller", str)  # pyright: ignore
        assert "Invalid installer implementation for 'TestInstaller'" in str(exc_info.value)
