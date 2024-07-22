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

from typing import Optional, Type

from .registry import Registry


class StrategyRegistry:
    """
    A registry for dynamically mapping and retrieving strategies based on system and template types.

    Allows strategies to be associated with specific combinations of system types and template types, facilitating the
    dynamic selection of strategies at runtime.
    """

    @classmethod
    def get_strategy(cls, strategy_interface: Type, system_type: Type, template_type: Type) -> Optional[Type]:
        """
        Retrieve a strategy from the registry based on system and template.

        Args:
            strategy_interface (Type): Interface type of strategy to retrieve.
            system_type (Type): The system type for retrieving the strategy.
            template_type (Type): The test template type for the strategy.

        Returns:
            Optional[Type]: The strategy class associated or None if not found.
        """
        registry = Registry()
        key = (strategy_interface, system_type, template_type)
        return registry.strategies_map.get(key)
