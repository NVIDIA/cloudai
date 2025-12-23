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


from cloudai.core import Registry
from cloudai.reporter import PerTestReporter, StatusReporter, TarballReporter
from cloudai.systems.kubernetes import KubernetesSystem
from cloudai.systems.lsf import LSFInstaller, LSFSystem
from cloudai.systems.runai import RunAIInstaller, RunAISystem
from cloudai.systems.slurm import SlurmInstaller, SlurmSystem
from cloudai.systems.standalone import StandaloneInstaller, StandaloneSystem
from cloudai.workloads.ai_dynamo import (
    AIDynamoKubernetesJsonGenStrategy,
    AIDynamoSlurmCommandGenStrategy,
    AIDynamoTestDefinition,
)
from cloudai.workloads.aiconfig import AiconfiguratorStandaloneCommandGenStrategy, AiconfiguratorTestDefinition
from cloudai.workloads.bash_cmd import BashCmdCommandGenStrategy, BashCmdTestDefinition
from cloudai.workloads.chakra_replay import (
    ChakraReplayGradingStrategy,
    ChakraReplaySlurmCommandGenStrategy,
    ChakraReplayTestDefinition,
)
from cloudai.workloads.ddlb import (
    DDLBTestDefinition,
    DDLBTestSlurmCommandGenStrategy,
)
from cloudai.workloads.deepep import (
    DeepEPSlurmCommandGenStrategy,
    DeepEPTestDefinition,
)
from cloudai.workloads.jax_toolbox import (
    GPTTestDefinition,
    GrokTestDefinition,
    JaxToolboxGradingStrategy,
    JaxToolboxSlurmCommandGenStrategy,
    NemotronTestDefinition,
)
from cloudai.workloads.megatron_bridge import (
    MegatronBridgeSlurmCommandGenStrategy,
    MegatronBridgeTestDefinition,
)
from cloudai.workloads.megatron_run import MegatronRunSlurmCommandGenStrategy, MegatronRunTestDefinition
from cloudai.workloads.nccl_test import (
    NcclComparisonReport,
    NCCLTestDefinition,
    NcclTestGradingStrategy,
    NcclTestKubernetesJsonGenStrategy,
    NcclTestRunAIJsonGenStrategy,
    NcclTestSlurmCommandGenStrategy,
)
from cloudai.workloads.nemo_launcher import (
    NeMoLauncherGradingStrategy,
    NeMoLauncherSlurmCommandGenStrategy,
    NeMoLauncherTestDefinition,
)
from cloudai.workloads.nemo_run import (
    NeMoRunSlurmCommandGenStrategy,
    NeMoRunTestDefinition,
)
from cloudai.workloads.nixl_bench import (
    NIXLBenchComparisonReport,
    NIXLBenchSlurmCommandGenStrategy,
    NIXLBenchTestDefinition,
)
from cloudai.workloads.nixl_kvbench import NIXLKVBenchSlurmCommandGenStrategy, NIXLKVBenchTestDefinition
from cloudai.workloads.nixl_perftest import NixlPerftestSlurmCommandGenStrategy, NixlPerftestTestDefinition
from cloudai.workloads.osu_bench import OSUBenchSlurmCommandGenStrategy, OSUBenchTestDefinition
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


CMD_GEN_STRATEGIES = {
    (SlurmSystem, ChakraReplayTestDefinition): ChakraReplaySlurmCommandGenStrategy,
    (SlurmSystem, DeepEPTestDefinition): DeepEPSlurmCommandGenStrategy,
    (SlurmSystem, GPTTestDefinition): JaxToolboxSlurmCommandGenStrategy,
    (SlurmSystem, GrokTestDefinition): JaxToolboxSlurmCommandGenStrategy,
    (SlurmSystem, NCCLTestDefinition): NcclTestSlurmCommandGenStrategy,
    (SlurmSystem, NeMoLauncherTestDefinition): NeMoLauncherSlurmCommandGenStrategy,
    (SlurmSystem, NeMoRunTestDefinition): NeMoRunSlurmCommandGenStrategy,
    (SlurmSystem, NemotronTestDefinition): JaxToolboxSlurmCommandGenStrategy,
    (SlurmSystem, SleepTestDefinition): SleepSlurmCommandGenStrategy,
    (SlurmSystem, SlurmContainerTestDefinition): SlurmContainerCommandGenStrategy,
    (SlurmSystem, UCCTestDefinition): UCCTestSlurmCommandGenStrategy,
    (SlurmSystem, DDLBTestDefinition): DDLBTestSlurmCommandGenStrategy,
    (SlurmSystem, MegatronRunTestDefinition): MegatronRunSlurmCommandGenStrategy,
    (SlurmSystem, MegatronBridgeTestDefinition): MegatronBridgeSlurmCommandGenStrategy,
    (StandaloneSystem, SleepTestDefinition): SleepStandaloneCommandGenStrategy,
    (StandaloneSystem, AiconfiguratorTestDefinition): AiconfiguratorStandaloneCommandGenStrategy,
    (LSFSystem, SleepTestDefinition): SleepLSFCommandGenStrategy,
    (SlurmSystem, TritonInferenceTestDefinition): TritonInferenceSlurmCommandGenStrategy,
    (SlurmSystem, NIXLBenchTestDefinition): NIXLBenchSlurmCommandGenStrategy,
    (SlurmSystem, AIDynamoTestDefinition): AIDynamoSlurmCommandGenStrategy,
    (SlurmSystem, BashCmdTestDefinition): BashCmdCommandGenStrategy,
    (SlurmSystem, NixlPerftestTestDefinition): NixlPerftestSlurmCommandGenStrategy,
    (SlurmSystem, NIXLKVBenchTestDefinition): NIXLKVBenchSlurmCommandGenStrategy,
    (SlurmSystem, OSUBenchTestDefinition): OSUBenchSlurmCommandGenStrategy,
}
JSON_GEN_STRATEGIES = {
    (KubernetesSystem, NCCLTestDefinition): NcclTestKubernetesJsonGenStrategy,
    (KubernetesSystem, SleepTestDefinition): SleepKubernetesJsonGenStrategy,
    (RunAISystem, NCCLTestDefinition): NcclTestRunAIJsonGenStrategy,
    (KubernetesSystem, AIDynamoTestDefinition): AIDynamoKubernetesJsonGenStrategy,
}
GRADING_STRATEGIES = {
    (SlurmSystem, ChakraReplayTestDefinition): ChakraReplayGradingStrategy,
    (SlurmSystem, GPTTestDefinition): JaxToolboxGradingStrategy,
    (SlurmSystem, GrokTestDefinition): JaxToolboxGradingStrategy,
    (SlurmSystem, NCCLTestDefinition): NcclTestGradingStrategy,
    (SlurmSystem, NeMoLauncherTestDefinition): NeMoLauncherGradingStrategy,
    (SlurmSystem, NemotronTestDefinition): JaxToolboxGradingStrategy,
    (SlurmSystem, SleepTestDefinition): SleepGradingStrategy,
    (SlurmSystem, UCCTestDefinition): UCCTestGradingStrategy,
}


def strategy2str(key: tuple) -> str:
    if len(key) == 2:
        return f"({key[0].__name__}, {key[1].__name__})"

    return f"({key[0].__name__}, {key[1].__name__}, {key[2].__name__})"


def test_grading_strategies():
    strategies = Registry().grading_strategies_map
    real = [strategy2str(k) for k in strategies]
    expected = [strategy2str(k) for k in GRADING_STRATEGIES]
    missing = set(expected) - set(real)
    extra = set(real) - set(expected)
    assert len(missing) == 0, f"Missing: {missing}"
    assert len(extra) == 0, f"Extra: {extra}"
    for key, value in GRADING_STRATEGIES.items():
        assert strategies[key] == value, f"Strategy {strategy2str(key)} is not {value}"


def test_command_gen_strategies():
    command_gen_strategies = Registry().command_gen_strategies_map
    real = [strategy2str(k) for k in command_gen_strategies]
    expected = [strategy2str(k) for k in CMD_GEN_STRATEGIES]
    missing = set(expected) - set(real)
    extra = set(real) - set(expected)
    assert len(missing) == 0, f"Missing: {missing}"
    assert len(extra) == 0, f"Extra: {extra}"
    for key, value in CMD_GEN_STRATEGIES.items():
        assert command_gen_strategies[key] == value, f"Command gen strategy {strategy2str(key)} is not {value}"


def test_json_gen_strategies():
    json_gen_strategies = Registry().json_gen_strategies_map
    real = [strategy2str(k) for k in json_gen_strategies]
    expected = [strategy2str(k) for k in JSON_GEN_STRATEGIES]
    missing = set(expected) - set(real)
    extra = set(real) - set(expected)
    assert len(missing) == 0, f"Missing: {missing}"
    assert len(extra) == 0, f"Extra: {extra}"
    for key, value in JSON_GEN_STRATEGIES.items():
        assert json_gen_strategies[key] == value, f"JSON gen strategy {strategy2str(key)} is not {value}"


def test_installers():
    installers = Registry().installers_map
    assert len(installers) == 5
    assert installers["standalone"] == StandaloneInstaller
    assert installers["slurm"] == SlurmInstaller
    assert installers["lsf"] == LSFInstaller
    assert installers["runai"] == RunAIInstaller


def test_definitions():
    test_defs = Registry().test_definitions_map
    assert len(test_defs) == 22
    for tdef in [
        ("UCCTest", UCCTestDefinition),
        ("DDLBTest", DDLBTestDefinition),
        ("NcclTest", NCCLTestDefinition),
        ("ChakraReplay", ChakraReplayTestDefinition),
        ("DeepEP", DeepEPTestDefinition),
        ("Sleep", SleepTestDefinition),
        ("NeMoLauncher", NeMoLauncherTestDefinition),
        ("NeMoRun", NeMoRunTestDefinition),
        ("JaxToolboxGPT", GPTTestDefinition),
        ("JaxToolboxGrok", GrokTestDefinition),
        ("JaxToolboxNemotron", NemotronTestDefinition),
        ("SlurmContainer", SlurmContainerTestDefinition),
        ("MegatronRun", MegatronRunTestDefinition),
        ("MegatronBridge", MegatronBridgeTestDefinition),
        ("TritonInference", TritonInferenceTestDefinition),
        ("NIXLBench", NIXLBenchTestDefinition),
        ("AIDynamo", AIDynamoTestDefinition),
        ("BashCmd", BashCmdTestDefinition),
        ("NixlPerftest", NixlPerftestTestDefinition),
        ("NIXLKVBench", NIXLKVBenchTestDefinition),
        ("Aiconfigurator", AiconfiguratorTestDefinition),
        ("OSUBench", OSUBenchTestDefinition),
    ]:
        assert test_defs[tdef[0]] == tdef[1]


def test_scenario_reports():
    scenario_reports = Registry().scenario_reports
    assert list(scenario_reports.keys()) == ["per_test", "status", "tarball", "nixl_bench_summary", "nccl_comparison"]
    assert list(scenario_reports.values()) == [
        PerTestReporter,
        StatusReporter,
        TarballReporter,
        NIXLBenchComparisonReport,
        NcclComparisonReport,
    ]


def test_report_configs():
    configs = Registry().report_configs
    assert list(configs.keys()) == ["per_test", "status", "tarball", "nixl_bench_summary", "nccl_comparison"]
    for name, rep_config in configs.items():
        assert rep_config.enable is True, f"Report {name} is not enabled by default"
