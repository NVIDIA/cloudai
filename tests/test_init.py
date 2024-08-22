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

import pytest
from cloudai import (
    GradingStrategy,
    InstallStrategy,
    JobIdRetrievalStrategy,
    JobSpecGenStrategy,
    Registry,
    ReportGenerationStrategy,
)
from cloudai.installer.kubernetes_installer import KubernetesInstaller
from cloudai.installer.slurm_installer import SlurmInstaller
from cloudai.installer.standalone_installer import StandaloneInstaller
from cloudai.schema.test_template.chakra_replay.grading_strategy import ChakraReplayGradingStrategy
from cloudai.schema.test_template.chakra_replay.report_generation_strategy import ChakraReplayReportGenerationStrategy
from cloudai.schema.test_template.chakra_replay.slurm_install_strategy import ChakraReplaySlurmInstallStrategy
from cloudai.schema.test_template.chakra_replay.slurm_job_spec_gen_strategy import ChakraReplaySlurmJobSpecGenStrategy
from cloudai.schema.test_template.chakra_replay.template import ChakraReplay
from cloudai.schema.test_template.common.slurm_job_id_retrieval_strategy import SlurmJobIdRetrievalStrategy
from cloudai.schema.test_template.common.standalone_job_id_retrieval_strategy import StandaloneJobIdRetrievalStrategy
from cloudai.schema.test_template.jax_toolbox.grading_strategy import JaxToolboxGradingStrategy
from cloudai.schema.test_template.jax_toolbox.report_generation_strategy import JaxToolboxReportGenerationStrategy
from cloudai.schema.test_template.jax_toolbox.slurm_install_strategy import JaxToolboxSlurmInstallStrategy
from cloudai.schema.test_template.jax_toolbox.slurm_job_spec_gen_strategy import JaxToolboxSlurmJobSpecGenStrategy
from cloudai.schema.test_template.jax_toolbox.template import JaxToolbox
from cloudai.schema.test_template.nccl_test.kubernetes_job_spec_gen_strategy import NcclTestKubernetesJobSpecGenStrategy
from cloudai.schema.test_template.nccl_test.kubernetes_report_generation_strategy import (
    KubernetesNcclTestReportGenerationStrategy,
)
from cloudai.schema.test_template.nccl_test.slurm_grading_strategy import SlurmNcclTestGradingStrategy
from cloudai.schema.test_template.nccl_test.slurm_install_strategy import NcclTestSlurmInstallStrategy
from cloudai.schema.test_template.nccl_test.slurm_job_spec_gen_strategy import NcclTestSlurmJobSpecGenStrategy
from cloudai.schema.test_template.nccl_test.slurm_report_generation_strategy import (
    SlurmNcclTestReportGenerationStrategy,
)
from cloudai.schema.test_template.nccl_test.template import NcclTest
from cloudai.schema.test_template.nemo_launcher.grading_strategy import NeMoLauncherGradingStrategy
from cloudai.schema.test_template.nemo_launcher.report_generation_strategy import NeMoLauncherReportGenerationStrategy
from cloudai.schema.test_template.nemo_launcher.slurm_install_strategy import NeMoLauncherSlurmInstallStrategy
from cloudai.schema.test_template.nemo_launcher.slurm_job_id_retrieval_strategy import (
    NeMoLauncherSlurmJobIdRetrievalStrategy,
)
from cloudai.schema.test_template.nemo_launcher.slurm_job_spec_gen_strategy import NeMoLauncherSlurmJobSpecGenStrategy
from cloudai.schema.test_template.nemo_launcher.template import NeMoLauncher
from cloudai.schema.test_template.sleep.grading_strategy import SleepGradingStrategy
from cloudai.schema.test_template.sleep.kubernetes_job_spec_gen_strategy import SleepKubernetesJobSpecGenStrategy
from cloudai.schema.test_template.sleep.report_generation_strategy import SleepReportGenerationStrategy
from cloudai.schema.test_template.sleep.slurm_job_spec_gen_strategy import SleepSlurmJobSpecGenStrategy
from cloudai.schema.test_template.sleep.standalone_install_strategy import SleepStandaloneInstallStrategy
from cloudai.schema.test_template.sleep.standalone_job_spec_gen_strategy import SleepStandaloneJobSpecGenStrategy
from cloudai.schema.test_template.sleep.template import Sleep
from cloudai.schema.test_template.ucc_test.grading_strategy import UCCTestGradingStrategy
from cloudai.schema.test_template.ucc_test.report_generation_strategy import UCCTestReportGenerationStrategy
from cloudai.schema.test_template.ucc_test.slurm_install_strategy import UCCTestSlurmInstallStrategy
from cloudai.schema.test_template.ucc_test.slurm_job_spec_gen_strategy import UCCTestSlurmJobSpecGenStrategy
from cloudai.schema.test_template.ucc_test.template import UCCTest
from cloudai.systems.kubernetes.kubernetes_system import KubernetesSystem
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.systems.standalone_system import StandaloneSystem


def test_system_parsers():
    parsers = Registry().system_parsers_map.keys()
    assert "standalone" in parsers
    assert "slurm" in parsers
    assert "kubernetes" in parsers
    assert len(parsers) == 3


def test_runners():
    runners = Registry().runners_map.keys()
    assert "standalone" in runners
    assert "slurm" in runners
    assert "kubernetes" in runners
    assert len(runners) == 3


@pytest.mark.parametrize(
    "key,value",
    [
        ((JobSpecGenStrategy, SlurmSystem, ChakraReplay), ChakraReplaySlurmJobSpecGenStrategy),
        ((JobSpecGenStrategy, SlurmSystem, JaxToolbox), JaxToolboxSlurmJobSpecGenStrategy),
        ((JobSpecGenStrategy, SlurmSystem, NcclTest), NcclTestSlurmJobSpecGenStrategy),
        ((JobSpecGenStrategy, SlurmSystem, NeMoLauncher), NeMoLauncherSlurmJobSpecGenStrategy),
        ((JobSpecGenStrategy, SlurmSystem, Sleep), SleepSlurmJobSpecGenStrategy),
        ((JobSpecGenStrategy, SlurmSystem, UCCTest), UCCTestSlurmJobSpecGenStrategy),
        ((JobSpecGenStrategy, StandaloneSystem, Sleep), SleepStandaloneJobSpecGenStrategy),
        ((GradingStrategy, SlurmSystem, ChakraReplay), ChakraReplayGradingStrategy),
        ((GradingStrategy, SlurmSystem, JaxToolbox), JaxToolboxGradingStrategy),
        ((GradingStrategy, SlurmSystem, NcclTest), SlurmNcclTestGradingStrategy),
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
        ((JobSpecGenStrategy, KubernetesSystem, NcclTest), NcclTestKubernetesJobSpecGenStrategy),
        ((JobSpecGenStrategy, KubernetesSystem, Sleep), SleepKubernetesJobSpecGenStrategy),
        ((ReportGenerationStrategy, KubernetesSystem, NcclTest), KubernetesNcclTestReportGenerationStrategy),
        ((ReportGenerationStrategy, SlurmSystem, ChakraReplay), ChakraReplayReportGenerationStrategy),
        ((ReportGenerationStrategy, SlurmSystem, JaxToolbox), JaxToolboxReportGenerationStrategy),
        ((ReportGenerationStrategy, SlurmSystem, NcclTest), SlurmNcclTestReportGenerationStrategy),
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
    assert len(installers) == 3
    assert installers["standalone"] == StandaloneInstaller
    assert installers["slurm"] == SlurmInstaller
    assert installers["kubernetes"] == KubernetesInstaller
