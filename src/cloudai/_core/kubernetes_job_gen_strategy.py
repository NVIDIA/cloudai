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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .test_scenario import TestRun
from .test_template_strategy import TestTemplateStrategy


@dataclass(frozen=True)
class JobStep:
    name: str
    command_type: str  # 'helm', 'kubectl', 'shell', 'port_forward', 'http'
    args: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    timeout: int = 300
    required: bool = True


@dataclass(frozen=True)
class JobSpec:
    steps: List[JobStep]
    manifest: Optional[Dict[str, Any]] = None


class KubernetesJobGenStrategy(TestTemplateStrategy, ABC):
    """Abstract base class for generating Kubernetes job specifications."""

    @abstractmethod
    def generate_spec(self, tr: TestRun) -> JobSpec:
        """
        Generate the job specification for a test run.

        Args:
            tr (TestRun): Contains the test and its run-specific configurations.

        Returns:
            JobSpec: The generated job specification containing steps and/or manifest.
        """
        pass
