#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from cloudai import (
    CommandGenStrategy,
    GradingStrategy,
    InstallStrategy,
    JobIdRetrievalStrategy,
    Registry,
    ReportGenerationStrategy,
)
from cloudai.installer.slurm_installer import SlurmInstaller
from cloudai.installer.standalone_installer import StandaloneInstaller
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
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.systems.standalone_system import StandaloneSystem


def test_system_parsers():
    parsers = Registry().system_parsers_map.keys()
    assert "standalone" in parsers
    assert "slurm" in parsers
    assert len(parsers) == 2


def test_runners():
    runners = Registry().runners_map.keys()
    assert "standalone" in runners
    assert "slurm" in runners
    assert len(runners) == 2


@pytest.mark.parametrize(
    "key,value",
    [
        ((CommandGenStrategy, SlurmSystem, ChakraReplay), ChakraReplaySlurmCommandGenStrategy),
        ((CommandGenStrategy, SlurmSystem, JaxToolbox), JaxToolboxSlurmCommandGenStrategy),
        ((CommandGenStrategy, SlurmSystem, NcclTest), NcclTestSlurmCommandGenStrategy),
        ((CommandGenStrategy, SlurmSystem, NeMoLauncher), NeMoLauncherSlurmCommandGenStrategy),
        ((CommandGenStrategy, SlurmSystem, Sleep), SleepStandaloneCommandGenStrategy),
        ((CommandGenStrategy, SlurmSystem, UCCTest), UCCTestSlurmCommandGenStrategy),
        ((CommandGenStrategy, StandaloneSystem, Sleep), SleepStandaloneCommandGenStrategy),
        ((GradingStrategy, SlurmSystem, ChakraReplay), ChakraReplayGradingStrategy),
        ((GradingStrategy, SlurmSystem, JaxToolbox), JaxToolboxGradingStrategy),
        ((GradingStrategy, SlurmSystem, NcclTest), NcclTestGradingStrategy),
        ((GradingStrategy, SlurmSystem, NeMoLauncher), NeMoLauncherGradingStrategy),
        ((GradingStrategy, SlurmSystem, Sleep), SleepGradingStrategy),
        ((GradingStrategy, SlurmSystem, UCCTest), UCCTestGradingStrategy),
        ((InstallStrategy, SlurmSystem, ChakraReplay), ChakraReplaySlurmInstallStrategy),
        ((InstallStrategy, SlurmSystem, JaxToolbox), JaxToolboxSlurmInstallStrategy),
        ((InstallStrategy, SlurmSystem, NcclTest), NcclTestSlurmInstallStrategy),
        ((InstallStrategy, SlurmSystem, NeMoLauncher), NeMoLauncherSlurmInstallStrategy),
        ((InstallStrategy, SlurmSystem, Sleep), SleepStandaloneInstallStrategy),
        ((InstallStrategy, SlurmSystem, UCCTest), UCCTestSlurmInstallStrategy),
        ((InstallStrategy, StandaloneSystem, Sleep), SleepStandaloneInstallStrategy),
        ((JobIdRetrievalStrategy, SlurmSystem, ChakraReplay), SlurmJobIdRetrievalStrategy),
        ((JobIdRetrievalStrategy, SlurmSystem, JaxToolbox), SlurmJobIdRetrievalStrategy),
        ((JobIdRetrievalStrategy, SlurmSystem, NcclTest), SlurmJobIdRetrievalStrategy),
        ((JobIdRetrievalStrategy, SlurmSystem, NeMoLauncher), NeMoLauncherSlurmJobIdRetrievalStrategy),
        ((JobIdRetrievalStrategy, SlurmSystem, UCCTest), SlurmJobIdRetrievalStrategy),
        ((JobIdRetrievalStrategy, StandaloneSystem, Sleep), StandaloneJobIdRetrievalStrategy),
        ((ReportGenerationStrategy, SlurmSystem, ChakraReplay), ChakraReplayReportGenerationStrategy),
        ((ReportGenerationStrategy, SlurmSystem, JaxToolbox), JaxToolboxReportGenerationStrategy),
        ((ReportGenerationStrategy, SlurmSystem, NcclTest), NcclTestReportGenerationStrategy),
        ((ReportGenerationStrategy, SlurmSystem, NeMoLauncher), NeMoLauncherReportGenerationStrategy),
        ((ReportGenerationStrategy, SlurmSystem, Sleep), SleepReportGenerationStrategy),
        ((ReportGenerationStrategy, SlurmSystem, UCCTest), UCCTestReportGenerationStrategy),
        ((ReportGenerationStrategy, StandaloneSystem, Sleep), SleepReportGenerationStrategy),
    ],
)
def test_strategies(key: tuple, value: type):
    strategies = Registry().strategies_map
    assert strategies[key] == value


def test_test_templates():
    test_templates = Registry().test_templates_map
    assert len(test_templates) == 6
    assert test_templates["ChakraReplay"] == ChakraReplay
    assert test_templates["JaxToolbox"] == JaxToolbox
    assert test_templates["NcclTest"] == NcclTest
    assert test_templates["NeMoLauncher"] == NeMoLauncher
    assert test_templates["Sleep"] == Sleep
    assert test_templates["UCCTest"] == UCCTest


def test_installers():
    installers = Registry().installers_map
    assert len(installers) == 2
    assert installers["standalone"] == StandaloneInstaller
    assert installers["slurm"] == SlurmInstaller
