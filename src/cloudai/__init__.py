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

from ._core.base_installer import BaseInstaller, InstallStatusResult
from ._core.base_job import BaseJob
from ._core.base_runner import BaseRunner
from ._core.base_system_parser import BaseSystemParser
from ._core.command_gen_strategy import CommandGenStrategy
from ._core.exceptions import JobIdRetrievalError
from ._core.grader import Grader
from ._core.grading_strategy import GradingStrategy
from ._core.install_strategy import InstallStrategy
from ._core.job_id_retrieval_strategy import JobIdRetrievalStrategy
from ._core.job_status_result import JobStatusResult
from ._core.job_status_retrieval_strategy import JobStatusRetrievalStrategy
from ._core.parser import Parser
from ._core.registry import Registry
from ._core.report_generation_strategy import ReportGenerationStrategy
from ._core.runner import Runner
from ._core.system import System
from ._core.test import Test
from ._core.test_scenario import TestScenario
from ._core.test_template import TestTemplate
from ._core.test_template_strategy import TestTemplateStrategy
from .installer.installer import Installer
from .installer.slurm_installer import SlurmInstaller
from .installer.standalone_installer import StandaloneInstaller
from .parser.system_parser.slurm_system_parser import SlurmSystemParser
from .parser.system_parser.standalone_system_parser import StandaloneSystemParser
from .report_generator import ReportGenerator
from .runner.slurm.slurm_runner import SlurmRunner
from .runner.standalone.standalone_runner import StandaloneRunner
from .schema.test_template.chakra_replay.grading_strategy import ChakraReplayGradingStrategy
from .schema.test_template.chakra_replay.report_generation_strategy import ChakraReplayReportGenerationStrategy
from .schema.test_template.chakra_replay.slurm_command_gen_strategy import ChakraReplaySlurmCommandGenStrategy
from .schema.test_template.chakra_replay.slurm_install_strategy import ChakraReplaySlurmInstallStrategy
from .schema.test_template.chakra_replay.template import ChakraReplay
from .schema.test_template.common.default_job_status_retrieval_strategy import DefaultJobStatusRetrievalStrategy
from .schema.test_template.common.slurm_job_id_retrieval_strategy import SlurmJobIdRetrievalStrategy
from .schema.test_template.common.standalone_job_id_retrieval_strategy import StandaloneJobIdRetrievalStrategy
from .schema.test_template.jax_toolbox.grading_strategy import JaxToolboxGradingStrategy
from .schema.test_template.jax_toolbox.job_status_retrieval_strategy import JaxToolboxJobStatusRetrievalStrategy
from .schema.test_template.jax_toolbox.report_generation_strategy import JaxToolboxReportGenerationStrategy
from .schema.test_template.jax_toolbox.slurm_command_gen_strategy import JaxToolboxSlurmCommandGenStrategy
from .schema.test_template.jax_toolbox.slurm_install_strategy import JaxToolboxSlurmInstallStrategy
from .schema.test_template.jax_toolbox.template import JaxToolbox
from .schema.test_template.nccl_test.grading_strategy import NcclTestGradingStrategy
from .schema.test_template.nccl_test.job_status_retrieval_strategy import NcclTestJobStatusRetrievalStrategy
from .schema.test_template.nccl_test.report_generation_strategy import NcclTestReportGenerationStrategy
from .schema.test_template.nccl_test.slurm_command_gen_strategy import NcclTestSlurmCommandGenStrategy
from .schema.test_template.nccl_test.slurm_install_strategy import NcclTestSlurmInstallStrategy
from .schema.test_template.nccl_test.template import NcclTest
from .schema.test_template.nemo_launcher.grading_strategy import NeMoLauncherGradingStrategy
from .schema.test_template.nemo_launcher.report_generation_strategy import NeMoLauncherReportGenerationStrategy
from .schema.test_template.nemo_launcher.slurm_command_gen_strategy import NeMoLauncherSlurmCommandGenStrategy
from .schema.test_template.nemo_launcher.slurm_install_strategy import NeMoLauncherSlurmInstallStrategy
from .schema.test_template.nemo_launcher.slurm_job_id_retrieval_strategy import (
    NeMoLauncherSlurmJobIdRetrievalStrategy,
)
from .schema.test_template.nemo_launcher.template import NeMoLauncher
from .schema.test_template.sleep.grading_strategy import SleepGradingStrategy
from .schema.test_template.sleep.report_generation_strategy import SleepReportGenerationStrategy
from .schema.test_template.sleep.slurm_command_gen_strategy import SleepSlurmCommandGenStrategy
from .schema.test_template.sleep.standalone_command_gen_strategy import SleepStandaloneCommandGenStrategy
from .schema.test_template.sleep.standalone_install_strategy import SleepStandaloneInstallStrategy
from .schema.test_template.sleep.template import Sleep
from .schema.test_template.ucc_test.grading_strategy import UCCTestGradingStrategy
from .schema.test_template.ucc_test.report_generation_strategy import UCCTestReportGenerationStrategy
from .schema.test_template.ucc_test.slurm_command_gen_strategy import UCCTestSlurmCommandGenStrategy
from .schema.test_template.ucc_test.slurm_install_strategy import UCCTestSlurmInstallStrategy
from .schema.test_template.ucc_test.template import UCCTest
from .systems.slurm.slurm_system import SlurmSystem
from .systems.standalone_system import StandaloneSystem

Registry().add_system_parser("standalone", StandaloneSystemParser)
Registry().add_system_parser("slurm", SlurmSystemParser)

Registry().add_runner("slurm", SlurmRunner)
Registry().add_runner("standalone", StandaloneRunner)

Registry().add_strategy(InstallStrategy, [SlurmSystem], [NcclTest], NcclTestSlurmInstallStrategy)
Registry().add_strategy(InstallStrategy, [SlurmSystem], [NeMoLauncher], NeMoLauncherSlurmInstallStrategy)
Registry().add_strategy(ReportGenerationStrategy, [SlurmSystem], [NcclTest], NcclTestReportGenerationStrategy)
Registry().add_strategy(CommandGenStrategy, [StandaloneSystem], [Sleep], SleepStandaloneCommandGenStrategy)
Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [Sleep], SleepSlurmCommandGenStrategy)
Registry().add_strategy(InstallStrategy, [SlurmSystem], [JaxToolbox], JaxToolboxSlurmInstallStrategy)
Registry().add_strategy(GradingStrategy, [SlurmSystem], [NcclTest], NcclTestGradingStrategy)
Registry().add_strategy(InstallStrategy, [SlurmSystem], [UCCTest], UCCTestSlurmInstallStrategy)
Registry().add_strategy(InstallStrategy, [StandaloneSystem, SlurmSystem], [Sleep], SleepStandaloneInstallStrategy)
Registry().add_strategy(
    ReportGenerationStrategy, [StandaloneSystem, SlurmSystem], [Sleep], SleepReportGenerationStrategy
)
Registry().add_strategy(ReportGenerationStrategy, [SlurmSystem], [NeMoLauncher], NeMoLauncherReportGenerationStrategy)
Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [NcclTest], NcclTestSlurmCommandGenStrategy)
Registry().add_strategy(GradingStrategy, [SlurmSystem], [Sleep], SleepGradingStrategy)
Registry().add_strategy(ReportGenerationStrategy, [SlurmSystem], [JaxToolbox], JaxToolboxReportGenerationStrategy)
Registry().add_strategy(JobIdRetrievalStrategy, [SlurmSystem], [NeMoLauncher], NeMoLauncherSlurmJobIdRetrievalStrategy)
Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [NeMoLauncher], NeMoLauncherSlurmCommandGenStrategy)
Registry().add_strategy(ReportGenerationStrategy, [SlurmSystem], [UCCTest], UCCTestReportGenerationStrategy)
Registry().add_strategy(GradingStrategy, [SlurmSystem], [NeMoLauncher], NeMoLauncherGradingStrategy)
Registry().add_strategy(GradingStrategy, [SlurmSystem], [JaxToolbox], JaxToolboxGradingStrategy)
Registry().add_strategy(GradingStrategy, [SlurmSystem], [UCCTest], UCCTestGradingStrategy)
Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [JaxToolbox], JaxToolboxSlurmCommandGenStrategy)
Registry().add_strategy(
    JobIdRetrievalStrategy,
    [SlurmSystem],
    [ChakraReplay, JaxToolbox, NcclTest, UCCTest, Sleep],
    SlurmJobIdRetrievalStrategy,
)
Registry().add_strategy(JobIdRetrievalStrategy, [StandaloneSystem], [Sleep], StandaloneJobIdRetrievalStrategy)
Registry().add_strategy(JobStatusRetrievalStrategy, [StandaloneSystem], [Sleep], DefaultJobStatusRetrievalStrategy)
Registry().add_strategy(JobStatusRetrievalStrategy, [SlurmSystem], [NcclTest], NcclTestJobStatusRetrievalStrategy)
Registry().add_strategy(JobStatusRetrievalStrategy, [SlurmSystem], [JaxToolbox], JaxToolboxJobStatusRetrievalStrategy)
Registry().add_strategy(
    JobStatusRetrievalStrategy,
    [SlurmSystem],
    [ChakraReplay, UCCTest, NeMoLauncher, Sleep],
    DefaultJobStatusRetrievalStrategy,
)
Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [UCCTest], UCCTestSlurmCommandGenStrategy)
Registry().add_strategy(InstallStrategy, [SlurmSystem], [ChakraReplay], ChakraReplaySlurmInstallStrategy)
Registry().add_strategy(ReportGenerationStrategy, [SlurmSystem], [ChakraReplay], ChakraReplayReportGenerationStrategy)
Registry().add_strategy(GradingStrategy, [SlurmSystem], [ChakraReplay], ChakraReplayGradingStrategy)
Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [ChakraReplay], ChakraReplaySlurmCommandGenStrategy)

Registry().add_test_template("ChakraReplay", ChakraReplay)
Registry().add_test_template("JaxToolbox", JaxToolbox)
Registry().add_test_template("NcclTest", NcclTest)
Registry().add_test_template("NeMoLauncher", NeMoLauncher)
Registry().add_test_template("Sleep", Sleep)
Registry().add_test_template("UCCTest", UCCTest)

Registry().add_installer("slurm", SlurmInstaller)
Registry().add_installer("standalone", StandaloneInstaller)

__all__ = [
    "BaseInstaller",
    "BaseJob",
    "BaseRunner",
    "BaseSystemParser",
    "CommandGenStrategy",
    "Grader",
    "GradingStrategy",
    "Installer",
    "InstallStatusResult",
    "InstallStrategy",
    "JobIdRetrievalError",
    "JobStatusResult",
    "Parser",
    "ReportGenerationStrategy",
    "ReportGenerator",
    "Runner",
    "System",
    "Test",
    "TestScenario",
    "TestTemplate",
    "TestTemplateStrategy",
]
