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
from typing import Callable, Type

from cloudai.schema.core import System, TestScenario

from .base_runner import BaseRunner


class Runner:
    """
    A wrapper class that creates and manages a specific runner instance
    based on the system's scheduler type.

    This class facilitates the initialization of the appropriate runner
    (StandaloneRunner or SlurmRunner) based on the scheduler specified in the
    system configuration.

    Attributes:
        _runners (Dict[str, Type[BaseRunner]]): A mapping from system
            types to their corresponding runner classes.
        runner (BaseRunner): The specific runner instance for the system.
        logger (logging.Logger): Logger for the runner activities.

    Args:
        mode (str): The operation mode ('dry-run', 'run').
        system (System): The system configuration object.
        test_scenario (TestScenario): The test scenario to be executed.
    """

    _runners = {}

    def __init__(self, mode: str, system: System, test_scenario: TestScenario):
        self.logger = logging.getLogger(__name__ + ".Runner")
        self.logger.info("Initializing Runner")
        self.runner = self.create_runner(mode, system, test_scenario)

    @classmethod
    def register(cls, system_type: str) -> Callable:
        """
        A decorator to register runner subclasses for specific system types.

        Args:
            system_type (str): The system type the runner can handle.

        Returns:
            Callable: A decorator function that registers the runner class.
        """

        def decorator(runner_class: Type[BaseRunner]) -> Type[BaseRunner]:
            if system_type in cls._runners:
                raise KeyError(f"Runner for {system_type} already registered.")
            cls._runners[system_type] = runner_class
            return runner_class

        return decorator

    def create_runner(self, mode: str, system: System, test_scenario: TestScenario) -> BaseRunner:
        """
        Dynamically create a runner instance based on the system's scheduler type.

        Args:
            mode (str): The operation mode ('dry-run', 'run').
            system (System): The system configuration.
            test_scenario (TestScenario): The test scenario to run.

        Returns:
            BaseRunner: A runner instance suitable for the system.

        Raises:
            NotImplementedError: If the scheduler type is not supported.
        """
        scheduler_type = system.scheduler.lower()
        runner_class = self._runners.get(scheduler_type)
        if runner_class is None:
            msg = f"No runner registered for scheduler: {scheduler_type}"
            self.logger.error(msg)
            raise NotImplementedError(msg)
        self.logger.info(f"Creating {runner_class.__name__}")
        return runner_class(mode, system, test_scenario)

    async def run(self):
        """
        Run the test scenario using the instantiated runner.
        """
        await self.runner.run()
