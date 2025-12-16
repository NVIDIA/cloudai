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

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ClassVar, List, Set, Tuple, Type

if TYPE_CHECKING:
    from ..configurator.base_agent import BaseAgent
    from ..models.scenario import ReportConfig
    from ..models.workload import TestDefinition
    from ..reporter import Reporter
    from .base_installer import BaseInstaller
    from .base_runner import BaseRunner
    from .command_gen_strategy import CommandGenStrategy
    from .grading_strategy import GradingStrategy
    from .json_gen_strategy import JsonGenStrategy
    from .report_generation_strategy import ReportGenerationStrategy
    from .system import System

RewardFunction = Callable[[List[float]], float]


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
    installers_map: ClassVar[dict[str, Type[BaseInstaller]]] = {}
    systems_map: ClassVar[dict[str, Type[System]]] = {}
    test_definitions_map: ClassVar[dict[str, Type[TestDefinition]]] = {}
    agents_map: ClassVar[dict[str, Type[BaseAgent]]] = {}
    reports_map: ClassVar[dict[Type[TestDefinition], Set[Type[ReportGenerationStrategy]]]] = {}
    scenario_reports: ClassVar[dict[str, type[Reporter]]] = {}
    report_configs: ClassVar[dict[str, ReportConfig]] = {}
    reward_functions_map: ClassVar[dict[str, RewardFunction]] = {}
    command_gen_strategies_map: ClassVar[dict[tuple[Type[System], Type[TestDefinition]], Type[CommandGenStrategy]]] = {}
    json_gen_strategies_map: ClassVar[dict[tuple[Type[System], Type[TestDefinition]], Type[JsonGenStrategy]]] = {}
    grading_strategies_map: ClassVar[dict[Tuple[Type[System], Type[TestDefinition]], Type[GradingStrategy]]] = {}

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
        """
        self.runners_map[name] = value

    def add_grading_strategy(
        self, system_type: Type[System], tdef_type: Type[TestDefinition], strategy: Type[GradingStrategy]
    ) -> None:
        key = (system_type, tdef_type)
        if key in self.grading_strategies_map:
            raise ValueError(f"Duplicating implementation for '{key}', use 'update()' for replacement.")
        self.update_grading_strategy(key, strategy)

    def update_grading_strategy(
        self, key: Tuple[Type[System], Type[TestDefinition]], value: Type[GradingStrategy]
    ) -> None:
        self.grading_strategies_map[key] = value

    def get_grading_strategy(self, system_type: Type[System], tdef_type: Type[TestDefinition]) -> Type[GradingStrategy]:
        if (system_type, tdef_type) not in self.grading_strategies_map:
            raise KeyError(f"Grading gen strategy for '{system_type.__name__}, {tdef_type.__name__}' not found.")
        return self.grading_strategies_map[(system_type, tdef_type)]

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
        """
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
        """
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
        """
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
        """
        self.agents_map[name] = value

    def add_report(self, tdef_type: Type[TestDefinition], value: Type[ReportGenerationStrategy]) -> None:
        existing_reports = self.reports_map.get(tdef_type, set())
        existing_reports.add(value)
        self.update_report(tdef_type, existing_reports)

    def update_report(self, tdef_type: Type[TestDefinition], reports: Set[Type[ReportGenerationStrategy]]) -> None:
        self.reports_map[tdef_type] = reports

    def add_scenario_report(self, name: str, report: type[Reporter], config: ReportConfig) -> None:
        if name in self.scenario_reports:
            raise ValueError(
                f"Duplicating scenario report implementation for '{name}', use 'update()' for replacement."
            )
        self.update_scenario_report(name, report, config)

    def update_scenario_report(self, name: str, report: type[Reporter], config: ReportConfig) -> None:
        self.scenario_reports[name] = report
        self.report_configs[name] = config

    def ordered_scenario_reports(self) -> list[tuple[str, type[Reporter]]]:
        def report_order(k: str) -> int:
            return {
                "per_test": 0,  # first
                "status": 2,
                "tarball": 3,  # last
            }.get(k, 1)

        return sorted(self.scenario_reports.items(), key=lambda kv: report_order(kv[0]))

    def add_reward_function(self, name: str, value: RewardFunction) -> None:
        if name in self.reward_functions_map:
            raise ValueError(f"Duplicating implementation for '{name}', use 'update()' for replacement.")
        self.update_reward_function(name, value)

    def update_reward_function(self, name: str, value: RewardFunction) -> None:
        self.reward_functions_map[name] = value

    def get_reward_function(self, name: str) -> RewardFunction:
        if name not in self.reward_functions_map:
            raise KeyError(
                f"Reward function '{name}' not found. Available functions: {list(self.reward_functions_map.keys())}"
            )
        return self.reward_functions_map[name]

    def add_command_gen_strategy(
        self, system_type: Type[System], tdef_type: Type[TestDefinition], value: Type[CommandGenStrategy]
    ) -> None:
        if (system_type, tdef_type) in self.command_gen_strategies_map:
            raise ValueError(
                f"Duplicating implementation for '{system_type.__name__}, {tdef_type.__name__}', use 'update()' "
                "for replacement."
            )
        self.update_command_gen_strategy(system_type, tdef_type, value)

    def update_command_gen_strategy(
        self, system_type: Type[System], tdef_type: Type[TestDefinition], value: Type[CommandGenStrategy]
    ) -> None:
        self.command_gen_strategies_map[(system_type, tdef_type)] = value

    def get_command_gen_strategy(
        self, system_type: Type[System], tdef_type: Type[TestDefinition]
    ) -> Type[CommandGenStrategy]:
        if (system_type, tdef_type) not in self.command_gen_strategies_map:
            raise KeyError(f"Command gen strategy for '{system_type.__name__}, {tdef_type.__name__}' not found.")
        return self.command_gen_strategies_map[(system_type, tdef_type)]

    def add_json_gen_strategy(
        self, system_type: Type[System], tdef_type: Type[TestDefinition], value: Type[JsonGenStrategy]
    ) -> None:
        if (system_type, tdef_type) in self.json_gen_strategies_map:
            raise ValueError(
                f"Duplicating implementation for '{system_type.__name__}, {tdef_type.__name__}', use 'update()' "
                "for replacement."
            )
        self.update_json_gen_strategy(system_type, tdef_type, value)

    def update_json_gen_strategy(
        self, system_type: Type[System], tdef_type: Type[TestDefinition], value: Type[JsonGenStrategy]
    ) -> None:
        self.json_gen_strategies_map[(system_type, tdef_type)] = value

    def get_json_gen_strategy(
        self, system_type: Type[System], tdef_type: Type[TestDefinition]
    ) -> Type[JsonGenStrategy]:
        if (system_type, tdef_type) not in self.json_gen_strategies_map:
            raise KeyError(f"JSON gen strategy for '{system_type.__name__}, {tdef_type.__name__}' not found.")
        return self.json_gen_strategies_map[(system_type, tdef_type)]
