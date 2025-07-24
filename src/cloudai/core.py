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

"""Core CloudAI base classes and interfaces."""

from ._core.base_installer import BaseInstaller, InstallStatusResult
from ._core.base_job import BaseJob
from ._core.base_reporter import Reporter, case_name
from ._core.base_runner import BaseRunner
from ._core.base_system_parser import BaseSystemParser
from ._core.command_gen_strategy import CommandGenStrategy
from ._core.exceptions import (
    JobIdRetrievalError,
    SystemConfigParsingError,
    TestConfigParsingError,
    TestScenarioParsingError,
    format_validation_error,
)
from ._core.grader import Grader
from ._core.grading_strategy import GradingStrategy
from ._core.installables import DockerImage, File, GitRepo, Installable, PythonExecutable
from ._core.job_status_result import JobStatusResult
from ._core.json_gen_strategy import JsonGenStrategy
from ._core.kubernetes_job_gen_strategy import KubernetesJobGenStrategy
from ._core.registry import Registry
from ._core.report_generation_strategy import ReportGenerationStrategy
from ._core.runner import Runner
from ._core.system import System
from ._core.test import Test
from ._core.test_scenario import METRIC_ERROR, TestDependency, TestRun, TestScenario
from ._core.test_template import TestTemplate
from ._core.test_template_strategy import TestTemplateStrategy
from .configurator.base_agent import BaseAgent
from .configurator.cloudai_gym import CloudAIGymEnv
from .configurator.grid_search import GridSearchAgent
from .models.workload import CmdArgs, NsysConfiguration, PredictorConfig, TestDefinition
from .parser import Parser
from .reporter import PerTestReporter, StatusReporter, TarballReporter
from .test_parser import TestParser
from .test_scenario_parser import TestScenarioParser

__all__ = [
    "METRIC_ERROR",
    "BaseAgent",
    "BaseInstaller",
    "BaseJob",
    "BaseRunner",
    "BaseSystemParser",
    "CloudAIGymEnv",
    "CmdArgs",
    "CommandGenStrategy",
    "DockerImage",
    "File",
    "GitRepo",
    "Grader",
    "GradingStrategy",
    "GridSearchAgent",
    "InstallStatusResult",
    "Installable",
    "JobIdRetrievalError",
    "JobStatusResult",
    "JsonGenStrategy",
    "KubernetesJobGenStrategy",
    "NsysConfiguration",
    "Parser",
    "PerTestReporter",
    "PredictorConfig",
    "PythonExecutable",
    "Registry",
    "ReportGenerationStrategy",
    "Reporter",
    "Runner",
    "StatusReporter",
    "System",
    "SystemConfigParsingError",
    "TarballReporter",
    "Test",
    "TestConfigParsingError",
    "TestDefinition",
    "TestDependency",
    "TestParser",
    "TestRun",
    "TestScenario",
    "TestScenarioParser",
    "TestScenarioParsingError",
    "TestTemplate",
    "TestTemplateStrategy",
    "case_name",
    "format_validation_error",
]
