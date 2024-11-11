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
    CommandGenStrategy,
    GradingStrategy,
    JobIdRetrievalStrategy,
    JsonGenStrategy,
    Registry,
    ReportGenerationStrategy,
)
from cloudai.installer.slurm_installer import SlurmInstaller
from cloudai.installer.standalone_installer import StandaloneInstaller
from cloudai.schema.test_template.chakra_replay.grading_strategy import ChakraReplayGradingStrategy
from cloudai.schema.test_template.chakra_replay.report_generation_strategy import ChakraReplayReportGenerationStrategy
from cloudai.schema.test_template.chakra_replay.slurm_command_gen_strategy import ChakraReplaySlurmCommandGenStrategy
from cloudai.schema.test_template.chakra_replay.template import ChakraReplay
from cloudai.schema.test_template.common.slurm_job_id_retrieval_strategy import SlurmJobIdRetrievalStrategy
from cloudai.schema.test_template.common.standalone_job_id_retrieval_strategy import StandaloneJobIdRetrievalStrategy
from cloudai.schema.test_template.jax_toolbox.grading_strategy import JaxToolboxGradingStrategy
from cloudai.schema.test_template.jax_toolbox.report_generation_strategy import JaxToolboxReportGenerationStrategy
from cloudai.schema.test_template.jax_toolbox.slurm_command_gen_strategy import JaxToolboxSlurmCommandGenStrategy
from cloudai.schema.test_template.nccl_test.grading_strategy import NcclTestGradingStrategy
from cloudai.schema.test_template.nccl_test.kubernetes_json_gen_strategy import NcclTestKubernetesJsonGenStrategy
from cloudai.schema.test_template.nccl_test.report_generation_strategy import NcclTestReportGenerationStrategy
from cloudai.schema.test_template.nccl_test.slurm_command_gen_strategy import NcclTestSlurmCommandGenStrategy
from cloudai.schema.test_template.nccl_test.template import NcclTest
from cloudai.schema.test_template.nemo_launcher.grading_strategy import NeMoLauncherGradingStrategy
from cloudai.schema.test_template.nemo_launcher.report_generation_strategy import NeMoLauncherReportGenerationStrategy
from cloudai.schema.test_template.nemo_launcher.slurm_command_gen_strategy import NeMoLauncherSlurmCommandGenStrategy
from cloudai.schema.test_template.nemo_launcher.slurm_job_id_retrieval_strategy import (
    NeMoLauncherSlurmJobIdRetrievalStrategy,
)
from cloudai.schema.test_template.nemo_launcher.template import NeMoLauncher
from cloudai.schema.test_template.sleep.grading_strategy import SleepGradingStrategy
from cloudai.schema.test_template.sleep.kubernetes_json_gen_strategy import SleepKubernetesJsonGenStrategy
from cloudai.schema.test_template.sleep.report_generation_strategy import SleepReportGenerationStrategy
from cloudai.schema.test_template.sleep.slurm_command_gen_strategy import SleepSlurmCommandGenStrategy
from cloudai.schema.test_template.sleep.standalone_command_gen_strategy import SleepStandaloneCommandGenStrategy
from cloudai.schema.test_template.sleep.template import Sleep
from cloudai.schema.test_template.ucc_test.grading_strategy import UCCTestGradingStrategy
from cloudai.schema.test_template.ucc_test.report_generation_strategy import UCCTestReportGenerationStrategy
from cloudai.schema.test_template.ucc_test.slurm_command_gen_strategy import UCCTestSlurmCommandGenStrategy
from cloudai.schema.test_template.ucc_test.template import UCCTest
from cloudai.systems.kubernetes.kubernetes_system import KubernetesSystem
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.systems.standalone_system import StandaloneSystem
from cloudai.test_definitions import (
    ChakraReplayTestDefinition,
    NCCLTestDefinition,
    NeMoLauncherTestDefinition,
    SleepTestDefinition,
    UCCTestDefinition,
)
from cloudai.test_definitions.gpt import GPTTestDefinition
from cloudai.test_definitions.grok import GrokTestDefinition
from cloudai.test_definitions.nemotron import NemotronTestDefinition


def test_systems():
    parsers = Registry().systems_map.keys()
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
        ((CommandGenStrategy, SlurmSystem, ChakraReplayTestDefinition), ChakraReplaySlurmCommandGenStrategy),
        ((CommandGenStrategy, SlurmSystem, GPTTestDefinition), JaxToolboxSlurmCommandGenStrategy),
        ((CommandGenStrategy, SlurmSystem, GrokTestDefinition), JaxToolboxSlurmCommandGenStrategy),
        ((CommandGenStrategy, SlurmSystem, NemotronTestDefinition), JaxToolboxSlurmCommandGenStrategy),
        ((CommandGenStrategy, SlurmSystem, NCCLTestDefinition), NcclTestSlurmCommandGenStrategy),
        ((CommandGenStrategy, SlurmSystem, NeMoLauncherTestDefinition), NeMoLauncherSlurmCommandGenStrategy),
        ((CommandGenStrategy, SlurmSystem, SleepTestDefinition), SleepSlurmCommandGenStrategy),
        ((CommandGenStrategy, SlurmSystem, UCCTestDefinition), UCCTestSlurmCommandGenStrategy),
        ((CommandGenStrategy, StandaloneSystem, SleepTestDefinition), SleepStandaloneCommandGenStrategy),
        ((GradingStrategy, SlurmSystem, ChakraReplayTestDefinition), ChakraReplayGradingStrategy),
        ((GradingStrategy, SlurmSystem, GPTTestDefinition), JaxToolboxGradingStrategy),
        ((GradingStrategy, SlurmSystem, GrokTestDefinition), JaxToolboxGradingStrategy),
        ((GradingStrategy, SlurmSystem, NemotronTestDefinition), JaxToolboxGradingStrategy),
        ((GradingStrategy, SlurmSystem, NCCLTestDefinition), NcclTestGradingStrategy),
        ((GradingStrategy, SlurmSystem, NeMoLauncherTestDefinition), NeMoLauncherGradingStrategy),
        ((GradingStrategy, SlurmSystem, SleepTestDefinition), SleepGradingStrategy),
        ((GradingStrategy, SlurmSystem, UCCTestDefinition), UCCTestGradingStrategy),
        ((JobIdRetrievalStrategy, SlurmSystem, ChakraReplayTestDefinition), SlurmJobIdRetrievalStrategy),
        ((JobIdRetrievalStrategy, SlurmSystem, GPTTestDefinition), SlurmJobIdRetrievalStrategy),
        ((JobIdRetrievalStrategy, SlurmSystem, GrokTestDefinition), SlurmJobIdRetrievalStrategy),
        ((JobIdRetrievalStrategy, SlurmSystem, NemotronTestDefinition), SlurmJobIdRetrievalStrategy),
        ((JobIdRetrievalStrategy, SlurmSystem, NCCLTestDefinition), SlurmJobIdRetrievalStrategy),
        ((JobIdRetrievalStrategy, SlurmSystem, NeMoLauncherTestDefinition), NeMoLauncherSlurmJobIdRetrievalStrategy),
        ((JobIdRetrievalStrategy, SlurmSystem, UCCTestDefinition), SlurmJobIdRetrievalStrategy),
        ((JobIdRetrievalStrategy, StandaloneSystem, SleepTestDefinition), StandaloneJobIdRetrievalStrategy),
        ((JsonGenStrategy, KubernetesSystem, NCCLTestDefinition), NcclTestKubernetesJsonGenStrategy),
        ((JsonGenStrategy, KubernetesSystem, SleepTestDefinition), SleepKubernetesJsonGenStrategy),
        ((ReportGenerationStrategy, SlurmSystem, ChakraReplayTestDefinition), ChakraReplayReportGenerationStrategy),
        ((ReportGenerationStrategy, SlurmSystem, GPTTestDefinition), JaxToolboxReportGenerationStrategy),
        ((ReportGenerationStrategy, SlurmSystem, GrokTestDefinition), JaxToolboxReportGenerationStrategy),
        ((ReportGenerationStrategy, SlurmSystem, NemotronTestDefinition), JaxToolboxReportGenerationStrategy),
        ((ReportGenerationStrategy, SlurmSystem, NCCLTestDefinition), NcclTestReportGenerationStrategy),
        ((ReportGenerationStrategy, KubernetesSystem, NCCLTestDefinition), NcclTestReportGenerationStrategy),
        ((ReportGenerationStrategy, SlurmSystem, NeMoLauncherTestDefinition), NeMoLauncherReportGenerationStrategy),
        ((ReportGenerationStrategy, SlurmSystem, SleepTestDefinition), SleepReportGenerationStrategy),
        ((ReportGenerationStrategy, SlurmSystem, UCCTestDefinition), UCCTestReportGenerationStrategy),
        ((ReportGenerationStrategy, StandaloneSystem, SleepTestDefinition), SleepReportGenerationStrategy),
    ],
)
def test_strategies(key: tuple, value: type):
    strategies = Registry().strategies_map
    assert strategies[key] == value


def test_test_templates():
    test_templates = Registry().test_templates_map
    assert len(test_templates) == 8
    assert test_templates["ChakraReplay"] == ChakraReplay
    assert test_templates["NcclTest"] == NcclTest
    assert test_templates["NeMoLauncher"] == NeMoLauncher
    assert test_templates["Sleep"] == Sleep
    assert test_templates["UCCTest"] == UCCTest


def test_installers():
    installers = Registry().installers_map
    assert len(installers) == 3
    assert installers["standalone"] == StandaloneInstaller
    assert installers["slurm"] == SlurmInstaller


def test_definitions():
    test_defs = Registry().test_definitions_map
    assert len(test_defs) == 8
    assert test_defs["UCCTest"] == UCCTestDefinition
    assert test_defs["NcclTest"] == NCCLTestDefinition
    assert test_defs["ChakraReplay"] == ChakraReplayTestDefinition
    assert test_defs["Sleep"] == SleepTestDefinition
    assert test_defs["NeMoLauncher"] == NeMoLauncherTestDefinition


def test_definitions_matches_templates():
    test_defs = Registry().test_definitions_map
    test_templates = Registry().test_templates_map

    def_names = set(test_defs.keys())
    template_names = set(test_templates.keys())
    assert def_names == template_names
