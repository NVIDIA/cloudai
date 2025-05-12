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

from ._core import (
    METRIC_ERROR,
    BaseAgent,
    BaseGym,
    BaseInstaller,
    BaseJob,
    BaseRunner,
    BaseSystemParser,
    CommandGenStrategy,
    DockerImage,
    File,
    GitRepo,
    Grader,
    GradingStrategy,
    Installable,
    InstallStatusResult,
    JobIdRetrievalStrategy,
    JobStatusResult,
    JobStatusRetrievalStrategy,
    JsonGenStrategy,
    PythonExecutable,
    Reporter,
    ReportGenerationStrategy,
    System,
    Test,
    TestDependency,
    TestRun,
    TestScenario,
    TestTemplate,
    TestTemplateStrategy,
)
from .configurator import GridSearchAgent
from .exceptions import (
    JobIdRetrievalError,
    SystemConfigParsingError,
    TestConfigParsingError,
    TestScenarioParsingError,
    format_validation_error,
)
from .registry import Registry
from .systems.kubernetes import KubernetesSystem
from .systems.lsf import LSFSystem
from .systems.runai import RunAISystem
from .systems.slurm import SlurmSystem
from .systems.standalone import StandaloneSystem
from .workloads.chakra_replay import ChakraReplayTestDefinition
from .workloads.jax_toolbox import GPTTestDefinition, GrokTestDefinition, NemotronTestDefinition
from .workloads.megatron_run import MegatronRunTestDefinition
from .workloads.nccl_test import NCCLTestDefinition
from .workloads.nemo_launcher import NeMoLauncherTestDefinition
from .workloads.nemo_run import NeMoRunTestDefinition
from .workloads.sleep import SleepTestDefinition
from .workloads.slurm_container import SlurmContainerTestDefinition
from .workloads.triton_inference import TritonInferenceTestDefinition
from .workloads.ucc_test import UCCTestDefinition

registry = Registry()
registry.add_agent("grid_search", GridSearchAgent)

registry.add_system("kubernetes", KubernetesSystem)
registry.add_system("lsf", LSFSystem)
registry.add_system("runai", RunAISystem)
registry.add_system("slurm", SlurmSystem)
registry.add_system("standalone", StandaloneSystem)

registry.add_test_definition("ChakraReplay", ChakraReplayTestDefinition)
registry.add_test_definition("JaxToolboxGrok", GrokTestDefinition)
registry.add_test_definition("JaxToolboxNemotron", NemotronTestDefinition)
registry.add_test_definition("JaxToolboxGPT", GPTTestDefinition)
registry.add_test_definition("MegatronRun", MegatronRunTestDefinition)
registry.add_test_definition("NcclTest", NCCLTestDefinition)
registry.add_test_definition("NeMoLauncher", NeMoLauncherTestDefinition)
registry.add_test_definition("NeMoRun", NeMoRunTestDefinition)
registry.add_test_definition("Sleep", SleepTestDefinition)
registry.add_test_definition("SlurmContainer", SlurmContainerTestDefinition)
registry.add_test_definition("TritonInference", TritonInferenceTestDefinition)
registry.add_test_definition("UCCTest", UCCTestDefinition)

__all__ = [
    "METRIC_ERROR",
    "BaseAgent",
    "BaseGym",
    "BaseInstaller",
    "BaseJob",
    "BaseRunner",
    "BaseSystemParser",
    "CommandGenStrategy",
    "DockerImage",
    "File",
    "GitRepo",
    "Grader",
    "GradingStrategy",
    "InstallStatusResult",
    "Installable",
    "JobIdRetrievalError",
    "JobIdRetrievalStrategy",
    "JobStatusResult",
    "JobStatusRetrievalStrategy",
    "JsonGenStrategy",
    "PythonExecutable",
    "ReportGenerationStrategy",
    "Reporter",
    "Reporter",
    "System",
    "SystemConfigParsingError",
    "Test",
    "TestConfigParsingError",
    "TestDependency",
    "TestRun",
    "TestScenario",
    "TestScenarioParsingError",
    "TestTemplate",
    "TestTemplateStrategy",
    "format_validation_error",
]
