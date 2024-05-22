from typing import Dict, List, Tuple, Type, Union

from cloudai.installer.base_installer import BaseInstaller
from cloudai.parser.core.base_system_parser import BaseSystemParser
from cloudai.runner.core.base_runner import BaseRunner
from cloudai.schema.core.strategy.command_gen_strategy import CommandGenStrategy
from cloudai.schema.core.strategy.grading_strategy import GradingStrategy
from cloudai.schema.core.strategy.install_strategy import InstallStrategy
from cloudai.schema.core.strategy.job_id_retrieval_strategy import JobIdRetrievalStrategy
from cloudai.schema.core.strategy.report_generation_strategy import ReportGenerationStrategy
from cloudai.schema.core.strategy.test_template_strategy import TestTemplateStrategy
from cloudai.schema.core.system import System
from cloudai.schema.core.test_template import TestTemplate


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
