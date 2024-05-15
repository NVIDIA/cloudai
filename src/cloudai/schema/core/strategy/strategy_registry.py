# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Callable, List, Optional, Type


class StrategyRegistry:
    """
    A registry for dynamically mapping and retrieving strategies based on system
    and template types. Allows strategies to be associated with specific
    combinations of system types and template types, facilitating the dynamic
    selection of strategies at runtime.
    """

    _registry = dict()

    @classmethod
    def register(
        cls,
        strategy_interface: Type,
        system_types: List[Type],
        template_types: List[Type],
        strategy: Type,
    ) -> None:
        """
        Registers a strategy for multiple system and template types.

        Args:
            strategy_interface (Type): Interface type of the strategy.
            system_types (List[Type]): Types of systems the strategy applies to.
            template_types (List[Type]): Types of test templates the strategy
                                         is for.
            strategy (Type): The strategy class to be registered.
        """
        for system_type in system_types:
            for template_type in template_types:
                key = (strategy_interface, system_type, template_type)
                cls._registry[key] = strategy

    @classmethod
    def get_strategy(cls, strategy_interface: Type, system_type: Type, template_type: Type) -> Optional[Type]:
        """
        Retrieves a strategy from the registry based on system and template.

        Args:
            strategy_interface (Type): Interface type of strategy to retrieve.
            system_type (Type): The system type for retrieving the strategy.
            template_type (Type): The test template type for the strategy.

        Returns:
            Optional[Type]: The strategy class associated or None if not found.
        """
        key = (strategy_interface, system_type, template_type)
        return cls._registry.get(key)

    @classmethod
    def strategy(
        cls,
        strategy_interface: Type,
        system_types: List[Type],
        template_types: List[Type],
    ) -> Callable[[Type], Type]:
        """
        Decorator for registering a strategy across multiple systems and
        templates.

        Args:
            strategy_interface (Type): Interface type of the strategy.
            system_types (List[Type]): System types the strategy applies to.
            template_types (List[Type]): Test template types for the strategy.

        Returns:
            Callable[[Type], Type]: Decorator for strategy class registration.
        """

        def decorator(strategy: Type) -> Type:
            cls.register(strategy_interface, system_types, template_types, strategy)
            return strategy

        return decorator
