import pytest
from cloudai._core.registry import Registry
from cloudai.parser.core.base_system_parser import BaseSystemParser


class MySystem(BaseSystemParser):
    pass


class AnotherSystem(BaseSystemParser):
    pass


@pytest.fixture
def registry():
    return Registry()


class TestRegistry:
    """This test verifies Registry class functionality.

    Since Registry is a Singleton, the order of cases is important.
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
