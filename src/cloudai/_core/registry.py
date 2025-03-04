# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import ClassVar, List, Tuple, Type, Union

from .base_installer import BaseInstaller
from .base_runner import BaseRunner
from .configurator.base_agent import BaseAgent
from .grading_strategy import GradingStrategy
from .job_id_retrieval_strategy import JobIdRetrievalStrategy
from .job_status_retrieval_strategy import JobStatusRetrievalStrategy
from .report_generation_strategy import ReportGenerationStrategy
from .system import System
from .test import TestDefinition
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

    runners_map: ClassVar[dict[str, Type[BaseRunner]]] = {}
    strategies_map: ClassVar[
        dict[
            Tuple[
                Type[
                    Union[
                        TestTemplateStrategy,
                        ReportGenerationStrategy,
                        JobIdRetrievalStrategy,
                        JobStatusRetrievalStrategy,
                        GradingStrategy,
                    ]
                ],
                Type[System],
                Type[TestDefinition],
            ],
            Type[
                Union[
                    TestTemplateStrategy,
                    ReportGenerationStrategy,
                    JobIdRetrievalStrategy,
                    JobStatusRetrievalStrategy,
                    GradingStrategy,
                ]
            ],
        ]
    ] = {}
    installers_map: ClassVar[dict[str, Type[BaseInstaller]]] = {}
    systems_map: ClassVar[dict[str, Type[System]]] = {}
    test_definitions_map: ClassVar[dict[str, Type[TestDefinition]]] = {}
    agents_map: ClassVar[dict[str, Type[BaseAgent]]] = {}

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
        strategy_interface: Type[
            Union[
                TestTemplateStrategy,
                ReportGenerationStrategy,
                JobIdRetrievalStrategy,
                JobStatusRetrievalStrategy,
                GradingStrategy,
            ]
        ],
        system_types: List[Type[System]],
        definition_types: List[Type[TestDefinition]],
        strategy: Type[
            Union[
                TestTemplateStrategy,
                ReportGenerationStrategy,
                JobIdRetrievalStrategy,
                JobStatusRetrievalStrategy,
                GradingStrategy,
            ]
        ],
    ) -> None:
        for system_type in system_types:
            for def_type in definition_types:
                key = (strategy_interface, system_type, def_type)
                if key in self.strategies_map:
                    raise ValueError(f"Duplicating implementation for '{key}', use 'update()' for replacement.")
                self.update_strategy(key, strategy)

    def update_strategy(
        self,
        key: Tuple[
            Type[
                Union[
                    TestTemplateStrategy,
                    ReportGenerationStrategy,
                    JobIdRetrievalStrategy,
                    JobStatusRetrievalStrategy,
                    GradingStrategy,
                ]
            ],
            Type[System],
            Type[TestDefinition],
        ],
        value: Type[
            Union[
                TestTemplateStrategy,
                ReportGenerationStrategy,
                JobIdRetrievalStrategy,
                JobStatusRetrievalStrategy,
                GradingStrategy,
            ]
        ],
    ) -> None:
        if not (
            issubclass(key[0], TestTemplateStrategy)
            or issubclass(key[0], ReportGenerationStrategy)
            or issubclass(key[0], JobIdRetrievalStrategy)
            or issubclass(key[0], JobStatusRetrievalStrategy)
            or issubclass(key[0], GradingStrategy)
        ):
            raise ValueError(
                "Invalid strategy interface type, should be subclass of 'TestTemplateStrategy' or "
                "'ReportGenerationStrategy' or 'JobIdRetrievalStrategy' or 'JobStatusRetrievalStrategy' or "
                "'GradingStrategy'."
            )
        if not issubclass(key[1], System):
            raise ValueError("Invalid system type, should be subclass of 'System'.")
        if not issubclass(key[2], TestDefinition):
            raise ValueError("Invalid test definition type, should be subclass of 'TestDefinition'.")

        if not (
            issubclass(value, TestTemplateStrategy)
            or issubclass(value, ReportGenerationStrategy)
            or issubclass(value, JobIdRetrievalStrategy)
            or issubclass(value, JobStatusRetrievalStrategy)
            or issubclass(value, GradingStrategy)
        ):
            raise ValueError(f"Invalid strategy implementation {value}, should be subclass of 'TestTemplateStrategy'.")
        self.strategies_map[key] = value

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

    def add_system(self, name: str, value: Type[System]) -> None:
        """
        Add a new system implementation mapping.

        Args:
            name (str): The name of the system.
            value (Type[System]): The system implementation.

        Raises:
            ValueError: If the system implementation already exists.
        """
        if name in self.systems_map:
            raise ValueError(f"Duplicating implementation for '{name}', use 'update()' for replacement.")
        self.update_system(name, value)

    def update_system(self, name: str, value: Type[System]) -> None:
        """
        Create or replace system implementation mapping.

        Args:
            name (str): The name of the system.
            value (Type[System]): The system implementation.

        Raises:
            ValueError: If value is not a subclass of System.
        """
        if not issubclass(value, System):
            raise ValueError(f"Invalid system implementation for '{name}', should be subclass of 'System'.")
        self.systems_map[name] = value

    def add_test_definition(self, name: str, value: Type[TestDefinition]) -> None:
        """
        Add a new test definition implementation mapping.

        Args:
            name (str): The name of the test definition.
            value (Type[TestDefinition]): The test definition implementation.

        Raises:
            ValueError: If the test definition implementation already exists.
        """
        if name in self.test_definitions_map:
            raise ValueError(f"Duplicating implementation for '{name}', use 'update()' for replacement.")
        self.update_test_definition(name, value)

    def update_test_definition(self, name: str, value: Type[TestDefinition]) -> None:
        """
        Create or replace test definition implementation mapping.

        Args:
            name (str): The name of the test definition.
            value (Type[TestDefinition]): The test definition implementation.

        Raises:
            ValueError: If value is not a subclass of TestDefinition.
        """
        if not issubclass(value, TestDefinition):
            raise ValueError(
                f"Invalid test definition implementation for '{name}', should be subclass of 'TestDefinition'."
            )
        self.test_definitions_map[name] = value

    def add_agent(self, name: str, value: Type[BaseAgent]) -> None:
        """
        Add a new agent implementation mapping.

        Args:
            name (str): The name of the agent.
            value (Type[BaseAgent]): The agent implementation.

        Raises:
            ValueError: If the agent implementation already exists.
        """
        if name in self.agents_map:
            raise ValueError(f"Duplicating implementation for '{name}', use 'update()' for replacement.")
        self.update_agent(name, value)

    def update_agent(self, name: str, value: Type[BaseAgent]) -> None:
        """
        Create or replace agent implementation mapping.

        Args:
            name (str): The name of the agent.
            value (Type[BaseAgent]): The agent implementation.

        Raises:
            ValueError: If value is not a subclass of BaseAgent.
        """
        if not issubclass(value, BaseAgent):
            raise ValueError(f"Invalid agent implementation for '{name}', should be subclass of 'BaseAgent'.")
        self.agents_map[name] = value
