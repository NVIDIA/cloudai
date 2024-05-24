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

import logging
from typing import Callable, Dict, Type

from cloudai._core.base_system_object_updater import BaseSystemObjectUpdater
from cloudai._core.system import System


class SystemObjectUpdater:
    """
    Facade class for updating system configurations based on the scheduler type.

    This class dynamically selects the appropriate updater subclass
    based on the provided system's scheduler type.

    Attributes
        _updaters (Dict[str, Type[BaseSystemObjectUpdater]]): A mapping from
            scheduler types to their corresponding updater subclasses. This
            allows the class to dynamically select the appropriate updater
            based on the system's scheduler.
    """

    _updaters = {}

    def __init__(self) -> None:
        """Initialize the SystemObjectUpdater instance, setting up the logger."""
        self.logger = logging.getLogger(__name__)

    @classmethod
    def register(cls, scheduler_type: str) -> Callable:
        """
        Register updater subclasses to handle specific scheduler types.

        To be used as a decorator.

        Args:
            scheduler_type (str): The scheduler type the updater can handle.

        Returns:
            Callable: A decorator function that registers the updater class
            to a scheduler type.
        """

        def decorator(
            updater_class: Type[BaseSystemObjectUpdater],
        ) -> Type[BaseSystemObjectUpdater]:
            if not issubclass(updater_class, BaseSystemObjectUpdater):
                raise ValueError(f"{updater_class.__name__} is not a subclass of BaseSystemObjectUpdater")
            cls._updaters[scheduler_type] = updater_class
            return updater_class

        return decorator

    @classmethod
    def get_supported_systems(cls) -> Dict[str, Type[BaseSystemObjectUpdater]]:
        """
        Retrieve the currently supported systems and their corresponding updater classes.

        Returns
            Dict[str, Type[BaseSystemObjectUpdater]]: A dictionary mapping
            scheduler types to their updater classes.
        """
        return cls._updaters

    def update(self, system: System) -> None:
        """
        Update the given system configuration using the appropriate updater.

        Based on the system's scheduler type.

        Args:
            system (System): The system configuration object to update.

        Raises:
            ValueError: If the system type is unsupported.
            NotImplementedError: If no updater is registered for the system's
            scheduler type.
        """
        scheduler = system.scheduler
        if scheduler not in self.get_supported_systems():
            raise ValueError(
                f"Unsupported system type '{scheduler}'. Supported types: "
                f"{', '.join(self.get_supported_systems().keys())}"
            )

        updater_class = self._updaters.get(scheduler)
        if updater_class is None:
            msg = f"Scheduler type '{scheduler}' not supported."
            self.logger.error(msg)
            raise NotImplementedError(msg)

        updater = updater_class()
        updater.update(system)
