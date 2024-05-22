import pytest
from cloudai._core.registry import Registry
from cloudai.parser.core.base_system_parser import BaseSystemParser
from cloudai.runner.core.base_runner import BaseRunner


class MySystem(BaseSystemParser):
    pass


class MyRunner(BaseRunner):
    pass


class AnotherSystem(BaseSystemParser):
    pass


class AnotherRunner(BaseRunner):
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
        registry.add_system_parser("system", MySystem)
        assert registry.system_parsers_map["system"] == MySystem

    def test_add_system_duplicate(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.add_system_parser("system", MySystem)
        assert "Duplicating implementation for 'system'" in str(exc_info.value)

    def test_update_system(self, registry: Registry):
        registry.update_system_parser("system", AnotherSystem)
        assert registry.system_parsers_map["system"] == AnotherSystem

    def test_invalid_type(self, registry: Registry):
        with pytest.raises(ValueError) as exc_info:
            registry.update_system_parser("TestSystem", str)  # pyright: ignore
        assert "Invalid system implementation for 'TestSystem'" in str(exc_info.value)


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
