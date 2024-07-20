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

import logging

from .base_runner import BaseRunner
from .registry import Registry
from .system import System
from .test_scenario import TestScenario


class Runner:
    """
    A wrapper class that creates and manages a specific runner instance based on the system's scheduler type.

    This class facilitates the initialization of the appropriate runner (StandaloneRunner or SlurmRunner) based on the
    scheduler specified in the system configuration.

    Attributes:
        _runners (Dict[str, Type[BaseRunner]]): A mapping from system types to their corresponding runner classes.
        runner (BaseRunner): The specific runner instance for the system.
        logger (logging.Logger): Logger for the runner activities.

    Args:
        mode (str): The operation mode ('dry-run', 'run').
        system (System): The system configuration object.
        test_scenario (TestScenario): The test scenario to be executed.
    """

    _runners = {}

    def __init__(self, mode: str, system: System, test_scenario: TestScenario):
        logging.info("Initializing Runner")
        self.runner = self.create_runner(mode, system, test_scenario)

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
        registry = Registry()
        scheduler_type = system.scheduler.lower()
        if scheduler_type not in registry.runners_map:
            msg = f"No runner registered for scheduler: {scheduler_type}"
            logging.error(msg)
            raise NotImplementedError(msg)
        runner_class = registry.runners_map[scheduler_type]
        logging.info(f"Creating {runner_class.__name__}")
        return runner_class(mode, system, test_scenario)

    async def run(self):
        """Run the test scenario using the instantiated runner."""
        await self.runner.run()
