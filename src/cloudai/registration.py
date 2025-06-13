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


def register_all():
    """Register all workloads, systems, runners, installers, and strategies."""
    from cloudai.configurator.grid_search import GridSearchAgent
    from cloudai.configurator.reward_functions import (
        identity_reward,
        inverse_reward,
        negative_reward,
    )
    from cloudai.core import (
        CommandGenStrategy,
        GradingStrategy,
        JobIdRetrievalStrategy,
        JobStatusRetrievalStrategy,
        JsonGenStrategy,
        Registry,
    )
    from cloudai.models.scenario import ReportConfig
    from cloudai.reporter import PerTestReporter, StatusReporter, TarballReporter

    # Import systems
    from cloudai.systems.kubernetes import KubernetesInstaller, KubernetesRunner, KubernetesSystem
    from cloudai.systems.lsf import LSFInstaller, LSFRunner, LSFSystem
    from cloudai.systems.runai import RunAIInstaller, RunAIRunner, RunAISystem
    from cloudai.systems.slurm import SlurmInstaller, SlurmRunner, SlurmSystem
    from cloudai.systems.standalone import StandaloneInstaller, StandaloneRunner, StandaloneSystem
    from cloudai.workloads.ai_dynamo import (
        AIDynamoReportGenerationStrategy,
        AIDynamoSlurmCommandGenStrategy,
        AIDynamoTestDefinition,
    )

    # Import all workloads and their strategies
    from cloudai.workloads.chakra_replay import (
        ChakraReplayGradingStrategy,
        ChakraReplayReportGenerationStrategy,
        ChakraReplaySlurmCommandGenStrategy,
        ChakraReplayTestDefinition,
    )

    # Import workload common strategies
    from cloudai.workloads.common import DefaultJobStatusRetrievalStrategy
    from cloudai.workloads.common.lsf_job_id_retrieval_strategy import LSFJobIdRetrievalStrategy
    from cloudai.workloads.common.slurm_job_id_retrieval_strategy import SlurmJobIdRetrievalStrategy
    from cloudai.workloads.common.standalone_job_id_retrieval_strategy import StandaloneJobIdRetrievalStrategy
    from cloudai.workloads.jax_toolbox import (
        GPTTestDefinition,
        GrokTestDefinition,
        JaxToolboxGradingStrategy,
        JaxToolboxJobStatusRetrievalStrategy,
        JaxToolboxReportGenerationStrategy,
        JaxToolboxSlurmCommandGenStrategy,
        NemotronTestDefinition,
    )
    from cloudai.workloads.megatron_run import (
        CheckpointTimingReportGenerationStrategy,
        MegatronRunSlurmCommandGenStrategy,
        MegatronRunTestDefinition,
    )
    from cloudai.workloads.nccl_test import (
        NCCLTestDefinition,
        NcclTestGradingStrategy,
        NcclTestJobStatusRetrievalStrategy,
        NcclTestKubernetesJsonGenStrategy,
        NcclTestPerformanceReportGenerationStrategy,
        NcclTestRunAIJsonGenStrategy,
        NcclTestSlurmCommandGenStrategy,
    )
    from cloudai.workloads.nemo_launcher import (
        NeMoLauncherGradingStrategy,
        NeMoLauncherReportGenerationStrategy,
        NeMoLauncherSlurmCommandGenStrategy,
        NeMoLauncherSlurmJobIdRetrievalStrategy,
        NeMoLauncherTestDefinition,
    )
    from cloudai.workloads.nemo_run import (
        NeMoRunDataStoreReportGenerationStrategy,
        NeMoRunJobStatusRetrievalStrategy,
        NeMoRunReportGenerationStrategy,
        NeMoRunSlurmCommandGenStrategy,
        NeMoRunTestDefinition,
    )
    from cloudai.workloads.nixl_bench import (
        NIXLBenchJobStatusRetrievalStrategy,
        NIXLBenchReportGenerationStrategy,
        NIXLBenchSlurmCommandGenStrategy,
        NIXLBenchTestDefinition,
    )
    from cloudai.workloads.sleep import (
        SleepGradingStrategy,
        SleepKubernetesJsonGenStrategy,
        SleepLSFCommandGenStrategy,
        SleepReportGenerationStrategy,
        SleepSlurmCommandGenStrategy,
        SleepStandaloneCommandGenStrategy,
        SleepTestDefinition,
    )
    from cloudai.workloads.slurm_container import (
        SlurmContainerCommandGenStrategy,
        SlurmContainerReportGenerationStrategy,
        SlurmContainerTestDefinition,
    )
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

    Registry().add_strategy(
        CommandGenStrategy, [StandaloneSystem], [SleepTestDefinition], SleepStandaloneCommandGenStrategy
    )
    Registry().add_strategy(CommandGenStrategy, [LSFSystem], [SleepTestDefinition], SleepLSFCommandGenStrategy)
    Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [SleepTestDefinition], SleepSlurmCommandGenStrategy)
    Registry().add_strategy(JsonGenStrategy, [KubernetesSystem], [SleepTestDefinition], SleepKubernetesJsonGenStrategy)
    Registry().add_strategy(
        JsonGenStrategy, [KubernetesSystem], [NCCLTestDefinition], NcclTestKubernetesJsonGenStrategy
    )
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
    Registry().add_strategy(
        CommandGenStrategy, [SlurmSystem], [NIXLBenchTestDefinition], NIXLBenchSlurmCommandGenStrategy
    )

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
            NIXLBenchTestDefinition,
            AIDynamoTestDefinition,
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
        [NeMoRunTestDefinition],
        NeMoRunJobStatusRetrievalStrategy,
    )
    Registry().add_strategy(
        JobStatusRetrievalStrategy,
        [SlurmSystem],
        [
            ChakraReplayTestDefinition,
            UCCTestDefinition,
            NeMoLauncherTestDefinition,
            SleepTestDefinition,
            SlurmContainerTestDefinition,
            MegatronRunTestDefinition,
            TritonInferenceTestDefinition,
            AIDynamoTestDefinition,
        ],
        DefaultJobStatusRetrievalStrategy,
    )
    Registry().add_strategy(
        JobStatusRetrievalStrategy,
        [SlurmSystem],
        [NIXLBenchTestDefinition],
        NIXLBenchJobStatusRetrievalStrategy,
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

    Registry().add_strategy(
        CommandGenStrategy, [SlurmSystem], [AIDynamoTestDefinition], AIDynamoSlurmCommandGenStrategy
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
    Registry().add_test_definition("NIXLBench", NIXLBenchTestDefinition)
    Registry().add_test_definition("AIDynamo", AIDynamoTestDefinition)

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
    Registry().add_report(NIXLBenchTestDefinition, NIXLBenchReportGenerationStrategy)
    Registry().add_report(AIDynamoTestDefinition, AIDynamoReportGenerationStrategy)

    Registry().add_scenario_report("per_test", PerTestReporter, ReportConfig(enable=True))
    Registry().add_scenario_report("status", StatusReporter, ReportConfig(enable=True))
    Registry().add_scenario_report("tarball", TarballReporter, ReportConfig(enable=True))

    Registry().add_reward_function("inverse", inverse_reward)
    Registry().add_reward_function("negative", negative_reward)
    Registry().add_reward_function("identity", identity_reward)
