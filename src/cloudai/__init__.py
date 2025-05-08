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

from ._core.base_installer import BaseInstaller, InstallStatusResult
from ._core.base_job import BaseJob
from ._core.base_runner import BaseRunner
from ._core.base_system_parser import BaseSystemParser
from ._core.command_gen_strategy import CommandGenStrategy
from ._core.configurator.base_agent import BaseAgent
from ._core.configurator.cloudai_gym import CloudAIGymEnv
from ._core.configurator.grid_search import GridSearchAgent
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
from ._core.job_id_retrieval_strategy import JobIdRetrievalStrategy
from ._core.job_status_result import JobStatusResult
from ._core.job_status_retrieval_strategy import JobStatusRetrievalStrategy
from ._core.json_gen_strategy import JsonGenStrategy
from ._core.registry import Registry
from ._core.report_generation_strategy import ReportGenerationStrategy
from ._core.reporter import PerTestReporter, Reporter, StatusReporter, TarballReporter
from ._core.runner import Runner
from ._core.system import System
from ._core.test import Test
from ._core.test_parser import TestParser
from ._core.test_scenario import TestRun, TestScenario
from ._core.test_scenario_parser import TestScenarioParser
from ._core.test_template import TestTemplate
from ._core.test_template_strategy import TestTemplateStrategy
from .installer.kubernetes_installer import KubernetesInstaller
from .installer.lsf_installer import LSFInstaller
from .installer.runai_installer import RunAIInstaller
from .installer.slurm_installer import SlurmInstaller
from .installer.standalone_installer import StandaloneInstaller
from .models.workload import CmdArgs, NsysConfiguration, PredictorConfig, TestDefinition
from .parser import Parser
from .runner.kubernetes.kubernetes_runner import KubernetesRunner
from .runner.lsf.lsf_runner import LSFRunner
from .runner.runai.runai_runner import RunAIRunner
from .runner.slurm.slurm_runner import SlurmRunner
from .runner.standalone.standalone_runner import StandaloneRunner
from .systems.kubernetes.kubernetes_system import KubernetesSystem
from .systems.lsf.lsf_system import LSFSystem
from .systems.runai.runai_system import RunAISystem
from .systems.slurm.slurm_system import SlurmSystem
from .systems.standalone_system import StandaloneSystem
from .workloads.chakra_replay import (
    ChakraReplayGradingStrategy,
    ChakraReplayReportGenerationStrategy,
    ChakraReplaySlurmCommandGenStrategy,
    ChakraReplayTestDefinition,
)
from .workloads.common import (
    DefaultJobStatusRetrievalStrategy,
    LSFJobIdRetrievalStrategy,
    SlurmJobIdRetrievalStrategy,
    StandaloneJobIdRetrievalStrategy,
)
from .workloads.jax_toolbox import (
    GPTTestDefinition,
    GrokTestDefinition,
    JaxToolboxGradingStrategy,
    JaxToolboxJobStatusRetrievalStrategy,
    JaxToolboxReportGenerationStrategy,
    JaxToolboxSlurmCommandGenStrategy,
    NemotronTestDefinition,
)
from .workloads.megatron_run import (
    CheckpointTimingReportGenerationStrategy,
    MegatronRunSlurmCommandGenStrategy,
    MegatronRunTestDefinition,
)
from .workloads.nccl_test import (
    NCCLTestDefinition,
    NcclTestGradingStrategy,
    NcclTestJobStatusRetrievalStrategy,
    NcclTestKubernetesJsonGenStrategy,
    NcclTestPerformanceReportGenerationStrategy,
    NcclTestRunAIJsonGenStrategy,
    NcclTestSlurmCommandGenStrategy,
)
from .workloads.nemo_launcher import (
    NeMoLauncherGradingStrategy,
    NeMoLauncherReportGenerationStrategy,
    NeMoLauncherSlurmCommandGenStrategy,
    NeMoLauncherSlurmJobIdRetrievalStrategy,
    NeMoLauncherTestDefinition,
)
from .workloads.nemo_run import (
    NeMoRunDataStoreReportGenerationStrategy,
    NeMoRunReportGenerationStrategy,
    NeMoRunSlurmCommandGenStrategy,
    NeMoRunTestDefinition,
)
from .workloads.sleep import (
    SleepGradingStrategy,
    SleepKubernetesJsonGenStrategy,
    SleepLSFCommandGenStrategy,
    SleepReportGenerationStrategy,
    SleepSlurmCommandGenStrategy,
    SleepStandaloneCommandGenStrategy,
    SleepTestDefinition,
)
from .workloads.slurm_container import (
    SlurmContainerCommandGenStrategy,
    SlurmContainerReportGenerationStrategy,
    SlurmContainerTestDefinition,
)
from .workloads.triton_inference import (
    TritonInferenceReportGenerationStrategy,
    TritonInferenceSlurmCommandGenStrategy,
    TritonInferenceTestDefinition,
)
from .workloads.ucc_test import (
    UCCTestDefinition,
    UCCTestGradingStrategy,
    UCCTestReportGenerationStrategy,
    UCCTestSlurmCommandGenStrategy,
)

Registry().add_runner("slurm", SlurmRunner)
Registry().add_runner("kubernetes", KubernetesRunner)
Registry().add_runner("standalone", StandaloneRunner)
Registry().add_runner("lsf", LSFRunner)
Registry().add_runner("runai", RunAIRunner)

Registry().add_strategy(
    CommandGenStrategy, [StandaloneSystem], [SleepTestDefinition], SleepStandaloneCommandGenStrategy
)
Registry().add_strategy(CommandGenStrategy, [LSFSystem], [SleepTestDefinition], SleepLSFCommandGenStrategy)
Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [SleepTestDefinition], SleepSlurmCommandGenStrategy)
Registry().add_strategy(JsonGenStrategy, [KubernetesSystem], [SleepTestDefinition], SleepKubernetesJsonGenStrategy)
Registry().add_strategy(JsonGenStrategy, [KubernetesSystem], [NCCLTestDefinition], NcclTestKubernetesJsonGenStrategy)
Registry().add_strategy(JsonGenStrategy, [RunAISystem], [NCCLTestDefinition], NcclTestRunAIJsonGenStrategy)
Registry().add_strategy(GradingStrategy, [SlurmSystem], [NCCLTestDefinition], NcclTestGradingStrategy)

Registry().add_strategy(
    CommandGenStrategy, [SlurmSystem], [MegatronRunTestDefinition], MegatronRunSlurmCommandGenStrategy
)
Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [NCCLTestDefinition], NcclTestSlurmCommandGenStrategy)
Registry().add_strategy(GradingStrategy, [SlurmSystem], [SleepTestDefinition], SleepGradingStrategy)

Registry().add_strategy(
    JobIdRetrievalStrategy, [SlurmSystem], [NeMoLauncherTestDefinition], NeMoLauncherSlurmJobIdRetrievalStrategy
)
Registry().add_strategy(
    CommandGenStrategy, [SlurmSystem], [NeMoLauncherTestDefinition], NeMoLauncherSlurmCommandGenStrategy
)
Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [NeMoRunTestDefinition], NeMoRunSlurmCommandGenStrategy)

Registry().add_strategy(GradingStrategy, [SlurmSystem], [NeMoLauncherTestDefinition], NeMoLauncherGradingStrategy)
Registry().add_strategy(
    GradingStrategy,
    [SlurmSystem],
    [GPTTestDefinition, GrokTestDefinition, NemotronTestDefinition],
    JaxToolboxGradingStrategy,
)
Registry().add_strategy(GradingStrategy, [SlurmSystem], [UCCTestDefinition], UCCTestGradingStrategy)
Registry().add_strategy(
    CommandGenStrategy,
    [SlurmSystem],
    [GPTTestDefinition, GrokTestDefinition, NemotronTestDefinition],
    JaxToolboxSlurmCommandGenStrategy,
)

Registry().add_strategy(
    JobIdRetrievalStrategy,
    [SlurmSystem],
    [
        ChakraReplayTestDefinition,
        GPTTestDefinition,
        GrokTestDefinition,
        NemotronTestDefinition,
        NCCLTestDefinition,
        UCCTestDefinition,
        SleepTestDefinition,
        NeMoRunTestDefinition,
        SlurmContainerTestDefinition,
        MegatronRunTestDefinition,
        TritonInferenceTestDefinition,
    ],
    SlurmJobIdRetrievalStrategy,
)
Registry().add_strategy(
    JobIdRetrievalStrategy, [StandaloneSystem], [SleepTestDefinition], StandaloneJobIdRetrievalStrategy
)
Registry().add_strategy(JobIdRetrievalStrategy, [LSFSystem], [SleepTestDefinition], LSFJobIdRetrievalStrategy)

Registry().add_strategy(
    JobStatusRetrievalStrategy,
    [KubernetesSystem],
    [SleepTestDefinition, NCCLTestDefinition],
    DefaultJobStatusRetrievalStrategy,
)
Registry().add_strategy(
    JobStatusRetrievalStrategy,
    [SlurmSystem],
    [GPTTestDefinition, GrokTestDefinition, NemotronTestDefinition],
    JaxToolboxJobStatusRetrievalStrategy,
)
Registry().add_strategy(
    JobStatusRetrievalStrategy,
    [SlurmSystem],
    [NCCLTestDefinition],
    NcclTestJobStatusRetrievalStrategy,
)
Registry().add_strategy(
    JobStatusRetrievalStrategy,
    [SlurmSystem],
    [
        ChakraReplayTestDefinition,
        UCCTestDefinition,
        NeMoLauncherTestDefinition,
        SleepTestDefinition,
        NeMoRunTestDefinition,
        SlurmContainerTestDefinition,
        MegatronRunTestDefinition,
        TritonInferenceTestDefinition,
    ],
    DefaultJobStatusRetrievalStrategy,
)
Registry().add_strategy(
    JobStatusRetrievalStrategy, [StandaloneSystem], [SleepTestDefinition], DefaultJobStatusRetrievalStrategy
)
Registry().add_strategy(
    JobStatusRetrievalStrategy, [LSFSystem], [SleepTestDefinition], DefaultJobStatusRetrievalStrategy
)
Registry().add_strategy(
    JobStatusRetrievalStrategy,
    [RunAISystem],
    [NCCLTestDefinition],
    DefaultJobStatusRetrievalStrategy,
)

Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [UCCTestDefinition], UCCTestSlurmCommandGenStrategy)

Registry().add_strategy(GradingStrategy, [SlurmSystem], [ChakraReplayTestDefinition], ChakraReplayGradingStrategy)
Registry().add_strategy(
    CommandGenStrategy, [SlurmSystem], [ChakraReplayTestDefinition], ChakraReplaySlurmCommandGenStrategy
)
Registry().add_strategy(
    CommandGenStrategy, [SlurmSystem], [SlurmContainerTestDefinition], SlurmContainerCommandGenStrategy
)
Registry().add_strategy(
    CommandGenStrategy, [SlurmSystem], [TritonInferenceTestDefinition], TritonInferenceSlurmCommandGenStrategy
)

Registry().add_installer("slurm", SlurmInstaller)
Registry().add_installer("standalone", StandaloneInstaller)
Registry().add_installer("kubernetes", KubernetesInstaller)
Registry().add_installer("lsf", LSFInstaller)
Registry().add_installer("runai", RunAIInstaller)

Registry().add_system("slurm", SlurmSystem)
Registry().add_system("standalone", StandaloneSystem)
Registry().add_system("kubernetes", KubernetesSystem)
Registry().add_system("lsf", LSFSystem)
Registry().add_system("runai", RunAISystem)

Registry().add_test_definition("UCCTest", UCCTestDefinition)
Registry().add_test_definition("NcclTest", NCCLTestDefinition)
Registry().add_test_definition("ChakraReplay", ChakraReplayTestDefinition)
Registry().add_test_definition("Sleep", SleepTestDefinition)
Registry().add_test_definition("NeMoLauncher", NeMoLauncherTestDefinition)
Registry().add_test_definition("NeMoRun", NeMoRunTestDefinition)
Registry().add_test_definition("JaxToolboxGPT", GPTTestDefinition)
Registry().add_test_definition("JaxToolboxGrok", GrokTestDefinition)
Registry().add_test_definition("JaxToolboxNemotron", NemotronTestDefinition)
Registry().add_test_definition("SlurmContainer", SlurmContainerTestDefinition)
Registry().add_test_definition("MegatronRun", MegatronRunTestDefinition)
Registry().add_test_definition("TritonInference", TritonInferenceTestDefinition)

Registry().add_agent("grid_search", GridSearchAgent)

Registry().add_report(ChakraReplayTestDefinition, ChakraReplayReportGenerationStrategy)
Registry().add_report(GPTTestDefinition, JaxToolboxReportGenerationStrategy)
Registry().add_report(GrokTestDefinition, JaxToolboxReportGenerationStrategy)
Registry().add_report(MegatronRunTestDefinition, CheckpointTimingReportGenerationStrategy)
Registry().add_report(NCCLTestDefinition, NcclTestPerformanceReportGenerationStrategy)
Registry().add_report(NeMoLauncherTestDefinition, NeMoLauncherReportGenerationStrategy)
Registry().add_report(NeMoRunTestDefinition, NeMoRunReportGenerationStrategy)
Registry().add_report(NeMoRunTestDefinition, NeMoRunDataStoreReportGenerationStrategy)
Registry().add_report(NemotronTestDefinition, JaxToolboxReportGenerationStrategy)
Registry().add_report(SleepTestDefinition, SleepReportGenerationStrategy)
Registry().add_report(SlurmContainerTestDefinition, SlurmContainerReportGenerationStrategy)
Registry().add_report(UCCTestDefinition, UCCTestReportGenerationStrategy)
Registry().add_report(TritonInferenceTestDefinition, TritonInferenceReportGenerationStrategy)

Registry().add_scenario_report(PerTestReporter)
Registry().add_scenario_report(StatusReporter)
Registry().add_scenario_report(TarballReporter)

__all__ = [
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
    "InstallStatusResult",
    "Installable",
    "JobIdRetrievalError",
    "JobStatusResult",
    "JsonGenStrategy",
    "NsysConfiguration",
    "Parser",
    "PredictorConfig",
    "PythonExecutable",
    "ReportGenerationStrategy",
    "Reporter",
    "RunAISystem",
    "Runner",
    "System",
    "SystemConfigParsingError",
    "Test",
    "TestConfigParsingError",
    "TestDefinition",
    "TestParser",
    "TestParser",
    "TestRun",
    "TestScenario",
    "TestScenarioParser",
    "TestScenarioParsingError",
    "TestTemplate",
    "TestTemplateStrategy",
    "format_validation_error",
]
