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

import cloudai.schema.test_template  # noqa
from cloudai.installer.slurm_installer import SlurmInstaller
from cloudai.installer.standalone_installer import StandaloneInstaller
from cloudai.parser.system_parser.slurm_system_parser import SlurmSystemParser
from cloudai.parser.system_parser.standalone_system_parser import StandaloneSystemParser
from cloudai.runner.slurm.slurm_runner import SlurmRunner
from cloudai.runner.standalone.standalone_runner import StandaloneRunner
from cloudai.schema.core.strategy.command_gen_strategy import CommandGenStrategy
from cloudai.schema.core.strategy.grading_strategy import GradingStrategy
from cloudai.schema.core.strategy.install_strategy import InstallStrategy
from cloudai.schema.core.strategy.job_id_retrieval_strategy import JobIdRetrievalStrategy
from cloudai.schema.core.strategy.report_generation_strategy import ReportGenerationStrategy
from cloudai.schema.system.slurm.slurm_system import SlurmSystem
from cloudai.schema.system.standalone_system import StandaloneSystem
from cloudai.schema.test_template.chakra_replay.grading_strategy import ChakraReplayGradingStrategy
from cloudai.schema.test_template.chakra_replay.report_generation_strategy import ChakraReplayReportGenerationStrategy
from cloudai.schema.test_template.chakra_replay.slurm_command_gen_strategy import ChakraReplaySlurmCommandGenStrategy
from cloudai.schema.test_template.chakra_replay.slurm_install_strategy import ChakraReplaySlurmInstallStrategy
from cloudai.schema.test_template.chakra_replay.template import ChakraReplay
from cloudai.schema.test_template.common.slurm_job_id_retrieval_strategy import SlurmJobIdRetrievalStrategy
from cloudai.schema.test_template.common.standalone_job_id_retrieval_strategy import StandaloneJobIdRetrievalStrategy
from cloudai.schema.test_template.jax_toolbox.grading_strategy import JaxToolboxGradingStrategy
from cloudai.schema.test_template.jax_toolbox.report_generation_strategy import JaxToolboxReportGenerationStrategy
from cloudai.schema.test_template.jax_toolbox.slurm_command_gen_strategy import JaxToolboxSlurmCommandGenStrategy
from cloudai.schema.test_template.jax_toolbox.slurm_install_strategy import JaxToolboxSlurmInstallStrategy
from cloudai.schema.test_template.jax_toolbox.template import JaxToolbox
from cloudai.schema.test_template.nccl_test.grading_strategy import NcclTestGradingStrategy
from cloudai.schema.test_template.nccl_test.report_generation_strategy import NcclTestReportGenerationStrategy
from cloudai.schema.test_template.nccl_test.slurm_command_gen_strategy import NcclTestSlurmCommandGenStrategy
from cloudai.schema.test_template.nccl_test.slurm_install_strategy import NcclTestSlurmInstallStrategy
from cloudai.schema.test_template.nccl_test.template import NcclTest
from cloudai.schema.test_template.nemo_launcher.grading_strategy import NeMoLauncherGradingStrategy
from cloudai.schema.test_template.nemo_launcher.report_generation_strategy import NeMoLauncherReportGenerationStrategy
from cloudai.schema.test_template.nemo_launcher.slurm_command_gen_strategy import NeMoLauncherSlurmCommandGenStrategy
from cloudai.schema.test_template.nemo_launcher.slurm_install_strategy import NeMoLauncherSlurmInstallStrategy
from cloudai.schema.test_template.nemo_launcher.slurm_job_id_retrieval_strategy import (
    NeMoLauncherSlurmJobIdRetrievalStrategy,
)
from cloudai.schema.test_template.nemo_launcher.template import NeMoLauncher
from cloudai.schema.test_template.sleep.grading_strategy import SleepGradingStrategy
from cloudai.schema.test_template.sleep.report_generation_strategy import SleepReportGenerationStrategy
from cloudai.schema.test_template.sleep.standalone_command_gen_strategy import SleepStandaloneCommandGenStrategy
from cloudai.schema.test_template.sleep.standalone_install_strategy import SleepStandaloneInstallStrategy
from cloudai.schema.test_template.sleep.template import Sleep
from cloudai.schema.test_template.ucc_test.grading_strategy import UCCTestGradingStrategy
from cloudai.schema.test_template.ucc_test.report_generation_strategy import UCCTestReportGenerationStrategy
from cloudai.schema.test_template.ucc_test.slurm_command_gen_strategy import UCCTestSlurmCommandGenStrategy
from cloudai.schema.test_template.ucc_test.slurm_install_strategy import UCCTestSlurmInstallStrategy
from cloudai.schema.test_template.ucc_test.template import UCCTest

from ._core.registry import Registry
from .grader import Grader
from .installer import Installer
from .parser.core.parser import Parser
from .report_generator import ReportGenerator
from .runner.core.runner import Runner
from .system_object_updater import SystemObjectUpdater

Registry().add_system_parser("standalone", StandaloneSystemParser)
Registry().add_system_parser("slurm", SlurmSystemParser)

Registry().add_runner("slurm", SlurmRunner)
Registry().add_runner("standalone", StandaloneRunner)

Registry().add_strategy(InstallStrategy, [SlurmSystem], [NcclTest], NcclTestSlurmInstallStrategy)
Registry().add_strategy(InstallStrategy, [SlurmSystem], [NeMoLauncher], NeMoLauncherSlurmInstallStrategy)
Registry().add_strategy(ReportGenerationStrategy, [SlurmSystem], [NcclTest], NcclTestReportGenerationStrategy)
Registry().add_strategy(CommandGenStrategy, [StandaloneSystem, SlurmSystem], [Sleep], SleepStandaloneCommandGenStrategy)
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
    JobIdRetrievalStrategy, [SlurmSystem], [ChakraReplay, JaxToolbox, NcclTest, UCCTest], SlurmJobIdRetrievalStrategy
)
Registry().add_strategy(JobIdRetrievalStrategy, [StandaloneSystem], [Sleep], StandaloneJobIdRetrievalStrategy)
Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [UCCTest], UCCTestSlurmCommandGenStrategy)
Registry().add_strategy(InstallStrategy, [SlurmSystem], [ChakraReplay], ChakraReplaySlurmInstallStrategy)
Registry().add_strategy(ReportGenerationStrategy, [SlurmSystem], [ChakraReplay], ChakraReplayReportGenerationStrategy)
Registry().add_strategy(GradingStrategy, [SlurmSystem], [ChakraReplay], ChakraReplayGradingStrategy)
Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [ChakraReplay], ChakraReplaySlurmCommandGenStrategy)


__all__ = [
    "Grader",
    "Installer",
    "Parser",
    "ReportGenerator",
    "Runner",
    "SystemObjectUpdater",
]
