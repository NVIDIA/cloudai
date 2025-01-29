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


from cloudai import (
    CommandGenStrategy,
    GradingStrategy,
    JobIdRetrievalStrategy,
    JobStatusRetrievalStrategy,
    JsonGenStrategy,
    Registry,
    ReportGenerationStrategy,
)
from cloudai.installer.slurm_installer import SlurmInstaller
from cloudai.installer.standalone_installer import StandaloneInstaller
from cloudai.schema.test_template.chakra_replay.grading_strategy import ChakraReplayGradingStrategy
from cloudai.schema.test_template.chakra_replay.report_generation_strategy import ChakraReplayReportGenerationStrategy
from cloudai.schema.test_template.chakra_replay.slurm_command_gen_strategy import ChakraReplaySlurmCommandGenStrategy
from cloudai.schema.test_template.common.default_job_status_retrieval_strategy import DefaultJobStatusRetrievalStrategy
from cloudai.schema.test_template.common.slurm_job_id_retrieval_strategy import SlurmJobIdRetrievalStrategy
from cloudai.schema.test_template.common.standalone_job_id_retrieval_strategy import StandaloneJobIdRetrievalStrategy
from cloudai.schema.test_template.jax_toolbox.grading_strategy import JaxToolboxGradingStrategy
from cloudai.schema.test_template.jax_toolbox.job_status_retrieval_strategy import JaxToolboxJobStatusRetrievalStrategy
from cloudai.schema.test_template.jax_toolbox.report_generation_strategy import JaxToolboxReportGenerationStrategy
from cloudai.schema.test_template.jax_toolbox.slurm_command_gen_strategy import JaxToolboxSlurmCommandGenStrategy
from cloudai.schema.test_template.nccl_test.grading_strategy import NcclTestGradingStrategy
from cloudai.schema.test_template.nccl_test.job_status_retrieval_strategy import NcclTestJobStatusRetrievalStrategy
from cloudai.schema.test_template.nccl_test.kubernetes_json_gen_strategy import NcclTestKubernetesJsonGenStrategy
from cloudai.schema.test_template.nccl_test.report_generation_strategy import NcclTestReportGenerationStrategy
from cloudai.schema.test_template.nccl_test.slurm_command_gen_strategy import NcclTestSlurmCommandGenStrategy
from cloudai.schema.test_template.nemo_launcher.grading_strategy import NeMoLauncherGradingStrategy
from cloudai.schema.test_template.nemo_launcher.report_generation_strategy import NeMoLauncherReportGenerationStrategy
from cloudai.schema.test_template.nemo_launcher.slurm_command_gen_strategy import NeMoLauncherSlurmCommandGenStrategy
from cloudai.schema.test_template.nemo_launcher.slurm_job_id_retrieval_strategy import (
    NeMoLauncherSlurmJobIdRetrievalStrategy,
)
from cloudai.schema.test_template.nemo_run.report_generation_strategy import NeMoRunReportGenerationStrategy
from cloudai.schema.test_template.nemo_run.slurm_command_gen_strategy import NeMoRunSlurmCommandGenStrategy
from cloudai.schema.test_template.sleep.grading_strategy import SleepGradingStrategy
from cloudai.schema.test_template.sleep.kubernetes_json_gen_strategy import SleepKubernetesJsonGenStrategy
from cloudai.schema.test_template.sleep.report_generation_strategy import SleepReportGenerationStrategy
from cloudai.schema.test_template.sleep.slurm_command_gen_strategy import SleepSlurmCommandGenStrategy
from cloudai.schema.test_template.sleep.standalone_command_gen_strategy import SleepStandaloneCommandGenStrategy
from cloudai.schema.test_template.slurm_container.report_generation_strategy import (
    SlurmContainerReportGenerationStrategy,
)
from cloudai.schema.test_template.slurm_container.slurm_command_gen_strategy import SlurmContainerCommandGenStrategy
from cloudai.schema.test_template.ucc_test.grading_strategy import UCCTestGradingStrategy
from cloudai.schema.test_template.ucc_test.report_generation_strategy import UCCTestReportGenerationStrategy
from cloudai.schema.test_template.ucc_test.slurm_command_gen_strategy import UCCTestSlurmCommandGenStrategy
from cloudai.systems.kubernetes.kubernetes_system import KubernetesSystem
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.systems.standalone_system import StandaloneSystem
from cloudai.test_definitions import (
    ChakraReplayTestDefinition,
    NCCLTestDefinition,
    NeMoLauncherTestDefinition,
    NeMoRunTestDefinition,
    SleepTestDefinition,
    UCCTestDefinition,
)
from cloudai.test_definitions.gpt import GPTTestDefinition
from cloudai.test_definitions.grok import GrokTestDefinition
from cloudai.test_definitions.nemotron import NemotronTestDefinition
from cloudai.test_definitions.slurm_container import SlurmContainerTestDefinition


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


ALL_STRATEGIES = {
    (CommandGenStrategy, SlurmSystem, ChakraReplayTestDefinition): ChakraReplaySlurmCommandGenStrategy,
    (CommandGenStrategy, SlurmSystem, GPTTestDefinition): JaxToolboxSlurmCommandGenStrategy,
    (CommandGenStrategy, SlurmSystem, GrokTestDefinition): JaxToolboxSlurmCommandGenStrategy,
    (CommandGenStrategy, SlurmSystem, NCCLTestDefinition): NcclTestSlurmCommandGenStrategy,
    (CommandGenStrategy, SlurmSystem, NeMoLauncherTestDefinition): NeMoLauncherSlurmCommandGenStrategy,
    (CommandGenStrategy, SlurmSystem, NeMoRunTestDefinition): NeMoRunSlurmCommandGenStrategy,
    (CommandGenStrategy, SlurmSystem, NemotronTestDefinition): JaxToolboxSlurmCommandGenStrategy,
    (CommandGenStrategy, SlurmSystem, SleepTestDefinition): SleepSlurmCommandGenStrategy,
    (CommandGenStrategy, SlurmSystem, SlurmContainerTestDefinition): SlurmContainerCommandGenStrategy,
    (CommandGenStrategy, SlurmSystem, UCCTestDefinition): UCCTestSlurmCommandGenStrategy,
    (CommandGenStrategy, StandaloneSystem, SleepTestDefinition): SleepStandaloneCommandGenStrategy,
    (GradingStrategy, SlurmSystem, ChakraReplayTestDefinition): ChakraReplayGradingStrategy,
    (GradingStrategy, SlurmSystem, GPTTestDefinition): JaxToolboxGradingStrategy,
    (GradingStrategy, SlurmSystem, GrokTestDefinition): JaxToolboxGradingStrategy,
    (GradingStrategy, SlurmSystem, NCCLTestDefinition): NcclTestGradingStrategy,
    (GradingStrategy, SlurmSystem, NeMoLauncherTestDefinition): NeMoLauncherGradingStrategy,
    (GradingStrategy, SlurmSystem, NemotronTestDefinition): JaxToolboxGradingStrategy,
    (GradingStrategy, SlurmSystem, SleepTestDefinition): SleepGradingStrategy,
    (GradingStrategy, SlurmSystem, UCCTestDefinition): UCCTestGradingStrategy,
    (JobIdRetrievalStrategy, SlurmSystem, ChakraReplayTestDefinition): SlurmJobIdRetrievalStrategy,
    (JobIdRetrievalStrategy, SlurmSystem, GPTTestDefinition): SlurmJobIdRetrievalStrategy,
    (JobIdRetrievalStrategy, SlurmSystem, GrokTestDefinition): SlurmJobIdRetrievalStrategy,
    (JobIdRetrievalStrategy, SlurmSystem, NCCLTestDefinition): SlurmJobIdRetrievalStrategy,
    (JobIdRetrievalStrategy, SlurmSystem, NeMoLauncherTestDefinition): NeMoLauncherSlurmJobIdRetrievalStrategy,
    (JobIdRetrievalStrategy, SlurmSystem, NeMoRunTestDefinition): SlurmJobIdRetrievalStrategy,
    (JobIdRetrievalStrategy, SlurmSystem, NemotronTestDefinition): SlurmJobIdRetrievalStrategy,
    (JobIdRetrievalStrategy, SlurmSystem, SleepTestDefinition): SlurmJobIdRetrievalStrategy,
    (JobIdRetrievalStrategy, SlurmSystem, SlurmContainerTestDefinition): SlurmJobIdRetrievalStrategy,
    (JobIdRetrievalStrategy, SlurmSystem, UCCTestDefinition): SlurmJobIdRetrievalStrategy,
    (JobIdRetrievalStrategy, StandaloneSystem, SleepTestDefinition): StandaloneJobIdRetrievalStrategy,
    (JobStatusRetrievalStrategy, KubernetesSystem, NCCLTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, KubernetesSystem, SleepTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, ChakraReplayTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, GPTTestDefinition): JaxToolboxJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, GrokTestDefinition): JaxToolboxJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, NCCLTestDefinition): NcclTestJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, NeMoLauncherTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, NeMoRunTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, NemotronTestDefinition): JaxToolboxJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, SleepTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, SlurmContainerTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, UCCTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, StandaloneSystem, SleepTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JsonGenStrategy, KubernetesSystem, NCCLTestDefinition): NcclTestKubernetesJsonGenStrategy,
    (JsonGenStrategy, KubernetesSystem, SleepTestDefinition): SleepKubernetesJsonGenStrategy,
    (ReportGenerationStrategy, KubernetesSystem, NCCLTestDefinition): NcclTestReportGenerationStrategy,
    (ReportGenerationStrategy, SlurmSystem, ChakraReplayTestDefinition): ChakraReplayReportGenerationStrategy,
    (ReportGenerationStrategy, SlurmSystem, GPTTestDefinition): JaxToolboxReportGenerationStrategy,
    (ReportGenerationStrategy, SlurmSystem, GrokTestDefinition): JaxToolboxReportGenerationStrategy,
    (ReportGenerationStrategy, SlurmSystem, NCCLTestDefinition): NcclTestReportGenerationStrategy,
    (ReportGenerationStrategy, SlurmSystem, NeMoLauncherTestDefinition): NeMoLauncherReportGenerationStrategy,
    (ReportGenerationStrategy, SlurmSystem, NeMoRunTestDefinition): NeMoRunReportGenerationStrategy,
    (ReportGenerationStrategy, SlurmSystem, NemotronTestDefinition): JaxToolboxReportGenerationStrategy,
    (ReportGenerationStrategy, SlurmSystem, SleepTestDefinition): SleepReportGenerationStrategy,
    (ReportGenerationStrategy, SlurmSystem, SlurmContainerTestDefinition): SlurmContainerReportGenerationStrategy,
    (ReportGenerationStrategy, SlurmSystem, UCCTestDefinition): UCCTestReportGenerationStrategy,
    (ReportGenerationStrategy, StandaloneSystem, SleepTestDefinition): SleepReportGenerationStrategy,
}


def test_strategies():
    def strategy2str(key: tuple) -> str:
        return f"({key[0].__name__}, {key[1].__name__}, {key[2].__name__})"

    strategies = Registry().strategies_map
    real = [strategy2str(k) for k in strategies]
    expected = [strategy2str(k) for k in ALL_STRATEGIES]
    missing = set(expected) - set(real)
    extra = set(real) - set(expected)
    assert len(missing) == 0, f"Missing: {missing}"
    assert len(extra) == 0, f"Extra: {extra}"
    for key, value in ALL_STRATEGIES.items():
        assert strategies[key] == value


def test_installers():
    installers = Registry().installers_map
    assert len(installers) == 3
    assert installers["standalone"] == StandaloneInstaller
    assert installers["slurm"] == SlurmInstaller


def test_definitions():
    test_defs = Registry().test_definitions_map
    assert len(test_defs) == 10
    for tdef in [
        ("UCCTest", UCCTestDefinition),
        ("NcclTest", NCCLTestDefinition),
        ("ChakraReplay", ChakraReplayTestDefinition),
        ("Sleep", SleepTestDefinition),
        ("NeMoLauncher", NeMoLauncherTestDefinition),
        ("NeMoRun", NeMoRunTestDefinition),
        ("JaxToolboxGPT", GPTTestDefinition),
        ("JaxToolboxGrok", GrokTestDefinition),
        ("JaxToolboxNemotron", NemotronTestDefinition),
        ("SlurmContainer", SlurmContainerTestDefinition),
    ]:
        assert test_defs[tdef[0]] == tdef[1]
