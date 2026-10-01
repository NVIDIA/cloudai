# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from ._core.base_installer import BaseInstaller
from ._core.base_job import BaseJob
from ._core.base_reporter import Reporter, case_name
from ._core.base_runner import BaseRunner
from ._core.base_system_parser import BaseSystemParser
from ._core.command_gen_strategy import CommandGenStrategy
from ._core.exceptions import (
    JobFailureError,
    JobIdRetrievalError,
    MissingTestError,
    SystemConfigParsingError,
    TestConfigParsingError,
    TestScenarioParsingError,
    format_validation_error,
)
from ._core.grader import Grader
from ._core.grading_strategy import GradingStrategy
from ._core.installables import (
    DockerImage,
    File,
    GitRepo,
    HFModel,
    Installable,
    InstallStatusResult,
    PythonEnvironment,
    PythonExecutable,
)
from ._core.job_status_result import JobStatusResult
from ._core.json_gen_strategy import JsonGenStrategy
from ._core.registry import Registry
from ._core.report_generation_strategy import ReportGenerationStrategy
from ._core.runner import Runner
from ._core.system import System
from ._core.test_scenario import (
    METRIC_ERROR,
    ConfigPaths,
    MetricErrorSentinel,
    MetricValue,
    TestDependency,
    TestRun,
    TestScenario,
)
from .configurator.base_agent import BaseAgent, BaseAgentConfig, RewardOverrides
from .configurator.cloudai_gym import CloudAIGymEnv
from .configurator.env_params import (
    CategoricalEncoding,
    Encoding,
    ObsLeafDescriptor,
    StructuredObservationProducer,
)
from .configurator.grid_search import GridSearchAgent
from .configurator.gymnasium_adapter import GymnasiumAdapter
from .models.workload import CmdArgs, NsysConfiguration, PredictorConfig, TestDefinition
from .parser import Parser
from .reporter import (
    JUnitReporter,
    PerTestReporter,
    ResultsUploadConfig,
    ResultsUploadReporter,
    StatusReporter,
    TarballReporter,
)
from .test_parser import TestParser
from .test_scenario_parser import TestScenarioParser
from .util.object_store import ObjectStore, S3ObjectStore, UploadStats

__all__ = [
    "METRIC_ERROR",
    "BaseAgent",
    "BaseAgentConfig",
    "BaseInstaller",
    "BaseJob",
    "BaseRunner",
    "BaseSystemParser",
    "CategoricalEncoding",
    "CloudAIGymEnv",
    "CmdArgs",
    "CommandGenStrategy",
    "ConfigPaths",
    "DockerImage",
    "Encoding",
    "File",
    "GitRepo",
    "Grader",
    "GradingStrategy",
    "GridSearchAgent",
    "GymnasiumAdapter",
    "HFModel",
    "InstallStatusResult",
    "Installable",
    "JUnitReporter",
    "JobFailureError",
    "JobIdRetrievalError",
    "JobStatusResult",
    "JsonGenStrategy",
    "MetricErrorSentinel",
    "MetricValue",
    "MissingTestError",
    "NsysConfiguration",
    "ObjectStore",
    "ObsLeafDescriptor",
    "Parser",
    "PerTestReporter",
    "PredictorConfig",
    "PythonEnvironment",
    "PythonExecutable",
    "Registry",
    "ReportGenerationStrategy",
    "Reporter",
    "ResultsUploadConfig",
    "ResultsUploadReporter",
    "RewardOverrides",
    "Runner",
    "S3ObjectStore",
    "StatusReporter",
    "StructuredObservationProducer",
    "System",
    "SystemConfigParsingError",
    "TarballReporter",
    "TestConfigParsingError",
    "TestDefinition",
    "TestDependency",
    "TestParser",
    "TestRun",
    "TestScenario",
    "TestScenarioParser",
    "TestScenarioParsingError",
    "UploadStats",
    "case_name",
    "format_validation_error",
]
