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


from cloudai import (
    CommandGenStrategy,
    GradingStrategy,
    JobIdRetrievalStrategy,
    JobStatusRetrievalStrategy,
    JsonGenStrategy,
    Registry,
)
from cloudai._core.reporter import PerTestReporter, StatusReporter, TarballReporter
from cloudai.installer.lsf_installer import LSFInstaller
from cloudai.installer.runai_installer import RunAIInstaller
from cloudai.installer.slurm_installer import SlurmInstaller
from cloudai.installer.standalone_installer import StandaloneInstaller
from cloudai.systems.kubernetes.kubernetes_system import KubernetesSystem
from cloudai.systems.lsf.lsf_system import LSFSystem
from cloudai.systems.runai.runai_system import RunAISystem
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.systems.standalone_system import StandaloneSystem
from cloudai.workloads.chakra_replay import (
    ChakraReplayGradingStrategy,
    ChakraReplaySlurmCommandGenStrategy,
    ChakraReplayTestDefinition,
)
from cloudai.workloads.common import SlurmJobIdRetrievalStrategy, StandaloneJobIdRetrievalStrategy
from cloudai.workloads.common.default_job_status_retrieval_strategy import DefaultJobStatusRetrievalStrategy
from cloudai.workloads.common.lsf_job_id_retrieval_strategy import LSFJobIdRetrievalStrategy
from cloudai.workloads.jax_toolbox import (
    GPTTestDefinition,
    GrokTestDefinition,
    JaxToolboxGradingStrategy,
    JaxToolboxJobStatusRetrievalStrategy,
    JaxToolboxSlurmCommandGenStrategy,
    NemotronTestDefinition,
)
from cloudai.workloads.megatron_run import MegatronRunSlurmCommandGenStrategy, MegatronRunTestDefinition
from cloudai.workloads.nccl_test import (
    NCCLTestDefinition,
    NcclTestGradingStrategy,
    NcclTestJobStatusRetrievalStrategy,
    NcclTestKubernetesJsonGenStrategy,
    NcclTestRunAIJsonGenStrategy,
    NcclTestSlurmCommandGenStrategy,
)
from cloudai.workloads.nemo_launcher import (
    NeMoLauncherGradingStrategy,
    NeMoLauncherSlurmCommandGenStrategy,
    NeMoLauncherSlurmJobIdRetrievalStrategy,
    NeMoLauncherTestDefinition,
)
from cloudai.workloads.nemo_run import (
    NeMoRunJobStatusRetrievalStrategy,
    NeMoRunSlurmCommandGenStrategy,
    NeMoRunTestDefinition,
)
from cloudai.workloads.nixl_bench.nixl_bench import NIXLBenchTestDefinition
from cloudai.workloads.nixl_bench.slurm_command_gen_strategy import NIXLBenchSlurmCommandGenStrategy
from cloudai.workloads.sleep import (
    SleepGradingStrategy,
    SleepKubernetesJsonGenStrategy,
    SleepLSFCommandGenStrategy,
    SleepSlurmCommandGenStrategy,
    SleepStandaloneCommandGenStrategy,
    SleepTestDefinition,
)
from cloudai.workloads.slurm_container import SlurmContainerCommandGenStrategy, SlurmContainerTestDefinition
from cloudai.workloads.triton_inference import (
    TritonInferenceSlurmCommandGenStrategy,
    TritonInferenceTestDefinition,
)
from cloudai.workloads.ucc_test import (
    UCCTestDefinition,
    UCCTestGradingStrategy,
    UCCTestSlurmCommandGenStrategy,
)


def test_systems():
    parsers = Registry().systems_map.keys()
    assert "standalone" in parsers
    assert "slurm" in parsers
    assert "kubernetes" in parsers
    assert "lsf" in parsers
    assert "runai" in parsers
    assert len(parsers) == 5


def test_runners():
    runners = Registry().runners_map.keys()
    assert "standalone" in runners
    assert "slurm" in runners
    assert "kubernetes" in runners
    assert "lsf" in runners
    assert "runai" in runners
    assert len(runners) == 5


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
    (CommandGenStrategy, SlurmSystem, MegatronRunTestDefinition): MegatronRunSlurmCommandGenStrategy,
    (CommandGenStrategy, StandaloneSystem, SleepTestDefinition): SleepStandaloneCommandGenStrategy,
    (CommandGenStrategy, LSFSystem, SleepTestDefinition): SleepLSFCommandGenStrategy,
    (CommandGenStrategy, SlurmSystem, TritonInferenceTestDefinition): TritonInferenceSlurmCommandGenStrategy,
    (CommandGenStrategy, SlurmSystem, NIXLBenchTestDefinition): NIXLBenchSlurmCommandGenStrategy,
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
    (JobIdRetrievalStrategy, SlurmSystem, MegatronRunTestDefinition): SlurmJobIdRetrievalStrategy,
    (JobIdRetrievalStrategy, SlurmSystem, TritonInferenceTestDefinition): SlurmJobIdRetrievalStrategy,
    (JobIdRetrievalStrategy, SlurmSystem, NIXLBenchTestDefinition): SlurmJobIdRetrievalStrategy,
    (JobIdRetrievalStrategy, StandaloneSystem, SleepTestDefinition): StandaloneJobIdRetrievalStrategy,
    (JobIdRetrievalStrategy, LSFSystem, SleepTestDefinition): LSFJobIdRetrievalStrategy,
    (JobStatusRetrievalStrategy, KubernetesSystem, NCCLTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, KubernetesSystem, SleepTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, ChakraReplayTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, GPTTestDefinition): JaxToolboxJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, GrokTestDefinition): JaxToolboxJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, NCCLTestDefinition): NcclTestJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, NeMoLauncherTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, NeMoRunTestDefinition): NeMoRunJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, NemotronTestDefinition): JaxToolboxJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, SleepTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, SlurmContainerTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, UCCTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, MegatronRunTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, TritonInferenceTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, SlurmSystem, NIXLBenchTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, StandaloneSystem, SleepTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, LSFSystem, SleepTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JobStatusRetrievalStrategy, RunAISystem, NCCLTestDefinition): DefaultJobStatusRetrievalStrategy,
    (JsonGenStrategy, KubernetesSystem, NCCLTestDefinition): NcclTestKubernetesJsonGenStrategy,
    (JsonGenStrategy, KubernetesSystem, SleepTestDefinition): SleepKubernetesJsonGenStrategy,
    (JsonGenStrategy, RunAISystem, NCCLTestDefinition): NcclTestRunAIJsonGenStrategy,
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
    assert len(installers) == 5
    assert installers["standalone"] == StandaloneInstaller
    assert installers["slurm"] == SlurmInstaller
    assert installers["lsf"] == LSFInstaller
    assert installers["runai"] == RunAIInstaller


def test_definitions():
    test_defs = Registry().test_definitions_map
    assert len(test_defs) == 13
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
        ("MegatronRun", MegatronRunTestDefinition),
        ("TritonInference", TritonInferenceTestDefinition),
        ("NIXLBench", NIXLBenchTestDefinition),
    ]:
        assert test_defs[tdef[0]] == tdef[1]


def test_scenario_reports():
    scenario_reports = Registry().scenario_reports
    assert len(scenario_reports) == 3
    assert PerTestReporter in scenario_reports
    assert StatusReporter in scenario_reports
    assert TarballReporter in scenario_reports
