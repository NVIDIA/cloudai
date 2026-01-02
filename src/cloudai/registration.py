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

import warnings
from importlib.metadata import entry_points


def register_entrypoint_agents():
    from cloudai.configurator.base_agent import BaseAgent
    from cloudai.core import Registry

    eps = entry_points(group="cloudai.agents")
    for ep in eps:
        cls = ep.load()
        if issubclass(cls, BaseAgent):
            Registry().add_agent(ep.name, cls)
        else:
            warnings.warn(
                f"Skipping entrypoint: {ep.name} -> {ep.value} class={cls} (not a subclass of BaseAgent)", stacklevel=2
            )


def register_all():
    """Register all workloads, systems, runners, installers, and strategies."""
    from cloudai.configurator.grid_search import GridSearchAgent
    from cloudai.configurator.reward_functions import (
        ai_dynamo_log_scale_reward,
        ai_dynamo_ratio_normalized_reward,
        ai_dynamo_weighted_normalized_reward,
        identity_reward,
        inverse_reward,
        negative_reward,
    )
    from cloudai.core import Registry
    from cloudai.models.scenario import ReportConfig
    from cloudai.reporter import PerTestReporter, StatusReporter, TarballReporter

    # Import systems
    from cloudai.systems.kubernetes import KubernetesInstaller, KubernetesRunner, KubernetesSystem
    from cloudai.systems.lsf import LSFInstaller, LSFRunner, LSFSystem
    from cloudai.systems.runai import RunAIInstaller, RunAIRunner, RunAISystem
    from cloudai.systems.slurm import SlurmInstaller, SlurmRunner, SlurmSystem
    from cloudai.systems.standalone import StandaloneInstaller, StandaloneRunner, StandaloneSystem
    from cloudai.workloads.ai_dynamo import (
        AIDynamoKubernetesJsonGenStrategy,
        AIDynamoReportGenerationStrategy,
        AIDynamoSlurmCommandGenStrategy,
        AIDynamoTestDefinition,
    )
    from cloudai.workloads.aiconfig import (
        AiconfiguratorReportGenerationStrategy,
        AiconfiguratorStandaloneCommandGenStrategy,
        AiconfiguratorTestDefinition,
    )

    # Import all workloads and their strategies
    from cloudai.workloads.bash_cmd.bash_cmd import BashCmdCommandGenStrategy, BashCmdTestDefinition
    from cloudai.workloads.chakra_replay import (
        ChakraReplayGradingStrategy,
        ChakraReplayReportGenerationStrategy,
        ChakraReplaySlurmCommandGenStrategy,
        ChakraReplayTestDefinition,
    )
    from cloudai.workloads.ddlb import (
        DDLBTestDefinition,
        DDLBTestSlurmCommandGenStrategy,
    )
    from cloudai.workloads.deepep import (
        DeepEPReportGenerationStrategy,
        DeepEPSlurmCommandGenStrategy,
        DeepEPTestDefinition,
    )
    from cloudai.workloads.jax_toolbox import (
        GPTTestDefinition,
        GrokTestDefinition,
        JaxToolboxGradingStrategy,
        JaxToolboxReportGenerationStrategy,
        JaxToolboxSlurmCommandGenStrategy,
        NemotronTestDefinition,
    )
    from cloudai.workloads.megatron_bridge import (
        MegatronBridgeReportGenerationStrategy,
        MegatronBridgeSlurmCommandGenStrategy,
        MegatronBridgeTestDefinition,
    )
    from cloudai.workloads.megatron_run import (
        CheckpointTimingReportGenerationStrategy,
        MegatronRunSlurmCommandGenStrategy,
        MegatronRunTestDefinition,
    )
    from cloudai.workloads.nccl_test import (
        ComparisonReportConfig,
        NcclComparisonReport,
        NCCLTestDefinition,
        NcclTestGradingStrategy,
        NcclTestKubernetesJsonGenStrategy,
        NcclTestPerformanceReportGenerationStrategy,
        NcclTestRunAIJsonGenStrategy,
        NcclTestSlurmCommandGenStrategy,
    )
    from cloudai.workloads.nemo_launcher import (
        NeMoLauncherGradingStrategy,
        NeMoLauncherReportGenerationStrategy,
        NeMoLauncherSlurmCommandGenStrategy,
        NeMoLauncherTestDefinition,
    )
    from cloudai.workloads.nemo_run import (
        NeMoRunDataStoreReportGenerationStrategy,
        NeMoRunReportGenerationStrategy,
        NeMoRunSlurmCommandGenStrategy,
        NeMoRunTestDefinition,
    )
    from cloudai.workloads.nixl_bench import (
        NIXLBenchComparisonReport,
        NIXLBenchReportGenerationStrategy,
        NIXLBenchSlurmCommandGenStrategy,
        NIXLBenchTestDefinition,
    )
    from cloudai.workloads.nixl_kvbench import NIXLKVBenchSlurmCommandGenStrategy, NIXLKVBenchTestDefinition
    from cloudai.workloads.nixl_perftest import (
        NIXLKVBenchDummyReport,
        NixlPerftestSlurmCommandGenStrategy,
        NixlPerftestTestDefinition,
    )
    from cloudai.workloads.osu_bench import (
        OSUBenchSlurmCommandGenStrategy,
        OSUBenchTestDefinition,
    )
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
        TritonInferenceReportGenerationStrategy,
        TritonInferenceSlurmCommandGenStrategy,
        TritonInferenceTestDefinition,
    )
    from cloudai.workloads.ucc_test import (
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

    Registry().add_json_gen_strategy(KubernetesSystem, SleepTestDefinition, SleepKubernetesJsonGenStrategy)
    Registry().add_json_gen_strategy(KubernetesSystem, NCCLTestDefinition, NcclTestKubernetesJsonGenStrategy)
    Registry().add_json_gen_strategy(KubernetesSystem, AIDynamoTestDefinition, AIDynamoKubernetesJsonGenStrategy)
    Registry().add_json_gen_strategy(RunAISystem, NCCLTestDefinition, NcclTestRunAIJsonGenStrategy)

    Registry().add_grading_strategy(SlurmSystem, NCCLTestDefinition, NcclTestGradingStrategy)
    Registry().add_grading_strategy(SlurmSystem, UCCTestDefinition, UCCTestGradingStrategy)
    Registry().add_grading_strategy(SlurmSystem, SleepTestDefinition, SleepGradingStrategy)
    Registry().add_grading_strategy(SlurmSystem, NeMoLauncherTestDefinition, NeMoLauncherGradingStrategy)
    Registry().add_grading_strategy(SlurmSystem, GPTTestDefinition, JaxToolboxGradingStrategy)
    Registry().add_grading_strategy(SlurmSystem, GrokTestDefinition, JaxToolboxGradingStrategy)
    Registry().add_grading_strategy(SlurmSystem, NemotronTestDefinition, JaxToolboxGradingStrategy)
    Registry().add_grading_strategy(SlurmSystem, ChakraReplayTestDefinition, ChakraReplayGradingStrategy)

    Registry().add_command_gen_strategy(StandaloneSystem, SleepTestDefinition, SleepStandaloneCommandGenStrategy)
    Registry().add_command_gen_strategy(LSFSystem, SleepTestDefinition, SleepLSFCommandGenStrategy)
    Registry().add_command_gen_strategy(SlurmSystem, SleepTestDefinition, SleepSlurmCommandGenStrategy)
    Registry().add_command_gen_strategy(
        StandaloneSystem, AiconfiguratorTestDefinition, AiconfiguratorStandaloneCommandGenStrategy
    )

    Registry().add_command_gen_strategy(SlurmSystem, MegatronRunTestDefinition, MegatronRunSlurmCommandGenStrategy)
    Registry().add_command_gen_strategy(SlurmSystem, NCCLTestDefinition, NcclTestSlurmCommandGenStrategy)
    Registry().add_command_gen_strategy(SlurmSystem, DDLBTestDefinition, DDLBTestSlurmCommandGenStrategy)
    Registry().add_command_gen_strategy(
        SlurmSystem, MegatronBridgeTestDefinition, MegatronBridgeSlurmCommandGenStrategy
    )

    Registry().add_command_gen_strategy(SlurmSystem, NeMoLauncherTestDefinition, NeMoLauncherSlurmCommandGenStrategy)
    Registry().add_command_gen_strategy(SlurmSystem, NeMoRunTestDefinition, NeMoRunSlurmCommandGenStrategy)
    Registry().add_command_gen_strategy(SlurmSystem, NIXLBenchTestDefinition, NIXLBenchSlurmCommandGenStrategy)

    Registry().add_command_gen_strategy(SlurmSystem, GPTTestDefinition, JaxToolboxSlurmCommandGenStrategy)
    Registry().add_command_gen_strategy(SlurmSystem, GrokTestDefinition, JaxToolboxSlurmCommandGenStrategy)
    Registry().add_command_gen_strategy(SlurmSystem, NemotronTestDefinition, JaxToolboxSlurmCommandGenStrategy)

    Registry().add_command_gen_strategy(SlurmSystem, UCCTestDefinition, UCCTestSlurmCommandGenStrategy)

    Registry().add_command_gen_strategy(SlurmSystem, ChakraReplayTestDefinition, ChakraReplaySlurmCommandGenStrategy)
    Registry().add_command_gen_strategy(SlurmSystem, DeepEPTestDefinition, DeepEPSlurmCommandGenStrategy)
    Registry().add_command_gen_strategy(SlurmSystem, SlurmContainerTestDefinition, SlurmContainerCommandGenStrategy)
    Registry().add_command_gen_strategy(
        SlurmSystem, TritonInferenceTestDefinition, TritonInferenceSlurmCommandGenStrategy
    )
    Registry().add_command_gen_strategy(SlurmSystem, NixlPerftestTestDefinition, NixlPerftestSlurmCommandGenStrategy)

    Registry().add_command_gen_strategy(SlurmSystem, AIDynamoTestDefinition, AIDynamoSlurmCommandGenStrategy)
    Registry().add_command_gen_strategy(SlurmSystem, BashCmdTestDefinition, BashCmdCommandGenStrategy)
    Registry().add_command_gen_strategy(SlurmSystem, NIXLKVBenchTestDefinition, NIXLKVBenchSlurmCommandGenStrategy)
    Registry().add_command_gen_strategy(SlurmSystem, OSUBenchTestDefinition, OSUBenchSlurmCommandGenStrategy)

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
    Registry().add_test_definition("DDLBTest", DDLBTestDefinition)
    Registry().add_test_definition("ChakraReplay", ChakraReplayTestDefinition)
    Registry().add_test_definition("DeepEP", DeepEPTestDefinition)
    Registry().add_test_definition("Sleep", SleepTestDefinition)
    Registry().add_test_definition("NeMoLauncher", NeMoLauncherTestDefinition)
    Registry().add_test_definition("NeMoRun", NeMoRunTestDefinition)
    Registry().add_test_definition("JaxToolboxGPT", GPTTestDefinition)
    Registry().add_test_definition("JaxToolboxGrok", GrokTestDefinition)
    Registry().add_test_definition("JaxToolboxNemotron", NemotronTestDefinition)
    Registry().add_test_definition("SlurmContainer", SlurmContainerTestDefinition)
    Registry().add_test_definition("MegatronRun", MegatronRunTestDefinition)
    Registry().add_test_definition("MegatronBridge", MegatronBridgeTestDefinition)
    Registry().add_test_definition("TritonInference", TritonInferenceTestDefinition)
    Registry().add_test_definition("NIXLBench", NIXLBenchTestDefinition)
    Registry().add_test_definition("AIDynamo", AIDynamoTestDefinition)
    Registry().add_test_definition("BashCmd", BashCmdTestDefinition)
    Registry().add_test_definition("NixlPerftest", NixlPerftestTestDefinition)
    Registry().add_test_definition("NIXLKVBench", NIXLKVBenchTestDefinition)
    Registry().add_test_definition("Aiconfigurator", AiconfiguratorTestDefinition)
    Registry().add_test_definition("OSUBench", OSUBenchTestDefinition)

    Registry().add_agent("grid_search", GridSearchAgent)

    Registry().add_report(ChakraReplayTestDefinition, ChakraReplayReportGenerationStrategy)
    Registry().add_report(DeepEPTestDefinition, DeepEPReportGenerationStrategy)
    Registry().add_report(GPTTestDefinition, JaxToolboxReportGenerationStrategy)
    Registry().add_report(GrokTestDefinition, JaxToolboxReportGenerationStrategy)
    Registry().add_report(MegatronRunTestDefinition, CheckpointTimingReportGenerationStrategy)
    Registry().add_report(MegatronBridgeTestDefinition, MegatronBridgeReportGenerationStrategy)
    Registry().add_report(NCCLTestDefinition, NcclTestPerformanceReportGenerationStrategy)
    Registry().add_report(NeMoLauncherTestDefinition, NeMoLauncherReportGenerationStrategy)
    Registry().add_report(NeMoRunTestDefinition, NeMoRunReportGenerationStrategy)
    Registry().add_report(NeMoRunTestDefinition, NeMoRunDataStoreReportGenerationStrategy)
    Registry().add_report(NemotronTestDefinition, JaxToolboxReportGenerationStrategy)
    Registry().add_report(UCCTestDefinition, UCCTestReportGenerationStrategy)
    Registry().add_report(TritonInferenceTestDefinition, TritonInferenceReportGenerationStrategy)
    Registry().add_report(NIXLBenchTestDefinition, NIXLBenchReportGenerationStrategy)
    Registry().add_report(AIDynamoTestDefinition, AIDynamoReportGenerationStrategy)
    Registry().add_report(AiconfiguratorTestDefinition, AiconfiguratorReportGenerationStrategy)
    Registry().add_report(NixlPerftestTestDefinition, NIXLKVBenchDummyReport)

    Registry().add_scenario_report("per_test", PerTestReporter, ReportConfig(enable=True))
    Registry().add_scenario_report("status", StatusReporter, ReportConfig(enable=True))
    Registry().add_scenario_report("tarball", TarballReporter, ReportConfig(enable=True))
    Registry().add_scenario_report(
        "nixl_bench_summary",
        NIXLBenchComparisonReport,
        ComparisonReportConfig(enable=True, group_by=["backend", "op_type"]),
    )
    Registry().add_scenario_report(
        "nccl_comparison", NcclComparisonReport, ComparisonReportConfig(enable=True, group_by=["subtest_name"])
    )

    Registry().add_reward_function("inverse", inverse_reward)
    Registry().add_reward_function("negative", negative_reward)
    Registry().add_reward_function("identity", identity_reward)
    Registry().add_reward_function("ai_dynamo_weighted_normalized", ai_dynamo_weighted_normalized_reward)
    Registry().add_reward_function("ai_dynamo_ratio_normalized", ai_dynamo_ratio_normalized_reward)
    Registry().add_reward_function("ai_dynamo_log_scale", ai_dynamo_log_scale_reward)

    register_entrypoint_agents()
