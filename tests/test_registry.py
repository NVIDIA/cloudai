import pytest
from cloudai import Registry
from cloudai.installer.base_installer import BaseInstaller
from cloudai.parser.core.base_system_parser import BaseSystemParser
from cloudai.runner.core.base_runner import BaseRunner
from cloudai.schema.core.strategy.job_id_retrieval_strategy import JobIdRetrievalStrategy
from cloudai.schema.core.strategy.report_generation_strategy import ReportGenerationStrategy
from cloudai.schema.core.strategy.test_template_strategy import TestTemplateStrategy
from cloudai.schema.core.system import System
from cloudai.schema.core.test_template import TestTemplate


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


class TestRegistry__StrategiesMap:
    """This test verifies Registry class functionality.

    Since Registry is a Singleton, the order of cases is important.
    Only covers the strategies_map attribute.
    """

    def test_add_strategy(self, registry: Registry):
        registry.add_strategy(MyStrategy, [MySystem], [MyTestTemplate], MyStrategy)
        registry.add_strategy(MyReportGenerationStrategy, [MySystem], [MyTestTemplate], MyReportGenerationStrategy)
        registry.add_strategy(MyJobIdRetrievalStrategy, [MySystem], [MyTestTemplate], MyJobIdRetrievalStrategy)

        assert registry.strategies_map[(MyStrategy, MySystem, MyTestTemplate)] == MyStrategy
        assert (
            registry.strategies_map[(MyReportGenerationStrategy, MySystem, MyTestTemplate)]
            == MyReportGenerationStrategy
        )
        assert registry.strategies_map[(MyJobIdRetrievalStrategy, MySystem, MyTestTemplate)] == MyJobIdRetrievalStrategy

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
            "'ReportGenerationStrategy' or 'JobIdRetrievalStrategy'."
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
