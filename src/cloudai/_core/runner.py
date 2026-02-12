# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import datetime
import logging
from types import FrameType
from typing import Optional

from .base_runner import BaseRunner
from .exceptions import JobFailureError
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

    def __init__(self, mode: str, system: System, test_scenario: TestScenario):
        logging.info(f"Initializing Runner [{mode.upper()}] mode")
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

        if not system.output_path.exists():
            system.output_path.mkdir()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_root = system.output_path / f"{test_scenario.name}_{current_time}"

        return runner_class(mode, system, test_scenario, results_root)

    def run(self):
        """Run the test scenario using the instantiated runner."""
        try:
            self.runner.run()
            logging.debug("All jobs finished successfully.")
        except JobFailureError as exc:
            logging.debug(f"Runner failed JobFailure exception: {exc}", exc_info=True)

    def cancel_on_signal(
        self,
        signum: int,
        frame: Optional[FrameType],  # noqa: Vulture
    ):
        logging.info(f"Signal {signum} received, shutting down...")
        self.runner.shutdown()
