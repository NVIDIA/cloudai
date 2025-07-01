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

from pathlib import Path
from typing import Any, Dict, Optional

from .grading_strategy import GradingStrategy
from .json_gen_strategy import JsonGenStrategy
from .system import System
from .test_scenario import TestRun


class TestTemplate:
    """
    Base class representing a test template.

    Providing a framework for test execution, including installation, uninstallation, and execution command generation
    based on system configurations and test parameters.
    """

    __test__ = False

    def __init__(self, system: System) -> None:
        """
        Initialize a TestTemplate instance.

        Args:
            system (System): System configuration for the test template.
        """
        self.system = system
        self._json_gen_strategy: Optional[JsonGenStrategy] = None
        self.grading_strategy: Optional[GradingStrategy] = None

    @property
    def json_gen_strategy(self) -> JsonGenStrategy:
        if self._json_gen_strategy is None:
            raise ValueError(
                "json_gen_strategy is missing. Ensure the strategy is registered in the Registry "
                "by calling the appropriate registration function for the system type."
            )
        return self._json_gen_strategy

    @json_gen_strategy.setter
    def json_gen_strategy(self, value: JsonGenStrategy) -> None:
        self._json_gen_strategy = value

    def gen_json(self, tr: TestRun) -> Dict[Any, Any]:
        """
        Generate a JSON string representing the Kubernetes job specification for this test using this template.

        Args:
            tr (TestRun): Contains the test and its run-specific configurations.

        Returns:
            Dict[Any, Any]: A dictionary representing the Kubernetes job specification.
        """
        if self.json_gen_strategy is None:
            raise ValueError(
                "json_gen_strategy is missing. Ensure the strategy is registered in the Registry "
                "by calling the appropriate registration function for the system type."
            )
        return self.json_gen_strategy.gen_json(tr)

    def grade(self, directory_path: Path, ideal_perf: float) -> Optional[float]:
        """
        Read the performance value from the directory.

        Args:
            directory_path (Path): Path to the directory containing performance data.
            ideal_perf (float): The ideal performance metric to compare against.

        Returns:
            Optional[float]: The performance value read from the directory.
        """
        if self.grading_strategy is not None:
            return self.grading_strategy.grade(directory_path, ideal_perf)
