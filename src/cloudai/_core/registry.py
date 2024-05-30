from typing import Dict, List, Tuple, Type, Union

from .base_installer import BaseInstaller
from .base_runner import BaseRunner
from .base_system_parser import BaseSystemParser
from .job_id_retrieval_strategy import JobIdRetrievalStrategy
from .report_generation_strategy import ReportGenerationStrategy
from .system import System
from .test_template import TestTemplate
from .test_template_strategy import TestTemplateStrategy


class Singleton(type):
    """Singleton metaclass."""

    _instance = None

    def __new__(cls, name, bases, dct):
        if not isinstance(cls._instance, cls):
            cls._instance = super().__new__(cls, name, bases, dct)
        return cls._instance


class Registry(metaclass=Singleton):
    """Registry for implementations mappings."""

    system_parsers_map: Dict[str, Type[BaseSystemParser]] = {}
    runners_map: Dict[str, Type[BaseRunner]] = {}
    strategies_map: Dict[
        Tuple[
            Type[Union[TestTemplateStrategy, ReportGenerationStrategy, JobIdRetrievalStrategy]],
            Type[System],
            Type[TestTemplate],
        ],
        Type[Union[TestTemplateStrategy, ReportGenerationStrategy, JobIdRetrievalStrategy]],
    ] = {}
    test_templates_map: Dict[str, Type[TestTemplate]] = {}
    installers_map: Dict[str, Type[BaseInstaller]] = {}

    def add_system_parser(self, name: str, value: Type[BaseSystemParser]) -> None:
        """
        Add a new system parser implementation mapping.

        Args:
            name (str): The name of the system parser.
            value (Type[BaseSystemParser]): The system parser implementation.

        Raises:
            ValueError: If the system parser implementation already exists.
        """
        if name in self.system_parsers_map:
            raise ValueError(f"Duplicating implementation for '{name}', use 'update()' for replacement.")
        self.update_system_parser(name, value)

    def update_system_parser(self, name: str, value: Type[BaseSystemParser]) -> None:
        """
        Create or replace system parser implementation mapping.

        Args:
            name (str): The name of the system parser.
            value (Type[BaseSystemParser]): The system parser implementation.

        Raises:
            ValueError: If value is not a subclass of BaseSystemParser.
        """
        if not issubclass(value, BaseSystemParser):
            raise ValueError(f"Invalid system implementation for '{name}', should be subclass of 'System'.")
        self.system_parsers_map[name] = value

    def add_runner(self, name: str, value: Type[BaseRunner]) -> None:
        """
        Add a new runner implementation mapping.

        Args:
            name (str): The name of the runner.
            value (Type[BaseRunner]): The runner implementation.

        Raises:
            ValueError: If the runner implementation already exists.
        """
        if name in self.runners_map:
            raise ValueError(f"Duplicating implementation for '{name}', use 'update()' for replacement.")
        self.update_runner(name, value)

    def update_runner(self, name: str, value: Type[BaseRunner]) -> None:
        """
        Create or replace runner implementation mapping.

        Args:
            name (str): The name of the runner.
            value (Type[BaseRunner]): The runner implementation.

        Raises:
            ValueError: If value is not a subclass of BaseRunner.
        """
        if not issubclass(value, BaseRunner):
            raise ValueError(f"Invalid runner implementation for '{name}', should be subclass of 'BaseRunner'.")
        self.runners_map[name] = value

    def add_strategy(
        self,
        strategy_interface: Type[Union[TestTemplateStrategy, ReportGenerationStrategy, JobIdRetrievalStrategy]],
        system_types: List[Type[System]],
        template_types: List[Type[TestTemplate]],
        strategy: Type[Union[TestTemplateStrategy, ReportGenerationStrategy, JobIdRetrievalStrategy]],
    ) -> None:
        for system_type in system_types:
            for template_type in template_types:
                key = (strategy_interface, system_type, template_type)
                if key in self.strategies_map:
                    raise ValueError(f"Duplicating implementation for '{key}', use 'update()' for replacement.")
                self.update_strategy(key, strategy)

    def update_strategy(
        self,
        key: Tuple[
            Type[Union[TestTemplateStrategy, ReportGenerationStrategy, JobIdRetrievalStrategy]],
            Type[System],
            Type[TestTemplate],
        ],
        value: Type[Union[TestTemplateStrategy, ReportGenerationStrategy, JobIdRetrievalStrategy]],
    ) -> None:
        if not (
            issubclass(key[0], TestTemplateStrategy)
            or issubclass(key[0], ReportGenerationStrategy)
            or issubclass(key[0], JobIdRetrievalStrategy)
        ):
            raise ValueError(
                "Invalid strategy interface type, should be subclass of 'TestTemplateStrategy' or "
                "'ReportGenerationStrategy' or 'JobIdRetrievalStrategy'."
            )
        if not issubclass(key[1], System):
            raise ValueError("Invalid system type, should be subclass of 'System'.")
        if not issubclass(key[2], TestTemplate):
            raise ValueError("Invalid test template type, should be subclass of 'TestTemplate'.")

        if not (
            issubclass(value, TestTemplateStrategy)
            or issubclass(value, ReportGenerationStrategy)
            or issubclass(value, JobIdRetrievalStrategy)
        ):
            raise ValueError(f"Invalid strategy implementation {value}, should be subclass of 'TestTemplateStrategy'.")
        self.strategies_map[key] = value

    def add_test_template(self, name: str, value: Type[TestTemplate]) -> None:
        """
        Add a new test template implementation mapping.

        Args:
            name (str): The name of the test template.
            value (Type[TestTemplate]): The test template implementation.

        Raises:
            ValueError: If the test template implementation already exists.
        """
        if name in self.test_templates_map:
            raise ValueError(f"Duplicating implementation for '{name}', use 'update()' for replacement.")
        self.update_test_template(name, value)

    def update_test_template(self, name: str, value: Type[TestTemplate]) -> None:
        """
        Create or replace test template implementation mapping.

        Args:
            name (str): The name of the test template.
            value (Type[TestTemplate]): The test template implementation.

        Raises:
            ValueError: If value is not a subclass of TestTemplate.
        """
        if not issubclass(value, TestTemplate):
            raise ValueError(
                f"Invalid test template implementation for '{name}', should be subclass of 'TestTemplate'."
            )
        self.test_templates_map[name] = value

    def add_installer(self, name: str, value: Type[BaseInstaller]) -> None:
        """
        Add a new installer implementation mapping.

        Args:
            name (str): The name of the installer.
            value (Type[BaseInstaller]): The installer implementation.

        Raises:
            ValueError: If the installer implementation already exists.
        """
        if name in self.installers_map:
            raise ValueError(f"Duplicating implementation for '{name}', use 'update()' for replacement.")
        self.update_installer(name, value)

    def update_installer(self, name: str, value: Type[BaseInstaller]) -> None:
        """
        Create or replace installer implementation mapping.

        Args:
            name (str): The name of the installer.
            value (Type[BaseInstaller]): The installer implementation.

        Raises:
            ValueError: If value is not a subclass of BaseInstaller.
        """
        if not issubclass(value, BaseInstaller):
            raise ValueError(f"Invalid installer implementation for '{name}', should be subclass of 'BaseInstaller'.")
        self.installers_map[name] = value
