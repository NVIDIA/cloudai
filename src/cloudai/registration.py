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
    from cloudai.core import (
        CommandGenStrategy,
        GradingStrategy,
        JobStatusRetrievalStrategy,
        JsonGenStrategy,
        Registry,
    )
    from cloudai.reporter import PerTestReporter, StatusReporter, TarballReporter

    # Import systems
    from cloudai.systems.kubernetes import KubernetesInstaller, KubernetesRunner, KubernetesSystem
    from cloudai.systems.lsf import LSFInstaller, LSFRunner, LSFSystem
    from cloudai.systems.runai import RunAIInstaller, RunAIRunner, RunAISystem
    from cloudai.systems.slurm import SlurmInstaller, SlurmRunner, SlurmSystem
    from cloudai.systems.standalone import StandaloneInstaller, StandaloneRunner, StandaloneSystem

    # Import all workloads and their strategies
    from cloudai.workloads.chakra_replay import (
        ChakraReplayGradingStrategy,
        ChakraReplayReportGenerationStrategy,
        ChakraReplaySlurmCommandGenStrategy,
        ChakraReplayTestDefinition,
    )

    # Import workload common strategies
    from cloudai.workloads.common import DefaultJobStatusRetrievalStrategy
    from cloudai.workloads.jax_toolbox import (
        GPTTestDefinition,
        GrokTestDefinition,
        JaxToolboxGradingStrategy,
        JaxToolboxJobStatusRetrievalStrategy,
        JaxToolboxReportGenerationStrategy,
        JaxToolboxSlurmCommandGenStrategy,
        JaxToolboxTestDefinition,
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

    registry = Registry()

    # Register scenario reporters
    registry.add_scenario_report(PerTestReporter)
    registry.add_scenario_report(StatusReporter)
    registry.add_scenario_report(TarballReporter)

    # Register systems
    registry.add_system("kubernetes", KubernetesSystem)
    registry.add_system("lsf", LSFSystem)
    registry.add_system("runai", RunAISystem)
    registry.add_system("slurm", SlurmSystem)
    registry.add_system("standalone", StandaloneSystem)

    # Register installers
    registry.add_installer("kubernetes", KubernetesInstaller)
    registry.add_installer("lsf", LSFInstaller)
    registry.add_installer("runai", RunAIInstaller)
    registry.add_installer("slurm", SlurmInstaller)
    registry.add_installer("standalone", StandaloneInstaller)

    # Register runners
    registry.add_runner("kubernetes", KubernetesRunner)
    registry.add_runner("lsf", LSFRunner)
    registry.add_runner("runai", RunAIRunner)
    registry.add_runner("slurm", SlurmRunner)
    registry.add_runner("standalone", StandaloneRunner)

    # Register test definitions
    registry.add_test_definition("ChakraReplay", ChakraReplayTestDefinition)
    registry.add_test_definition("JaxToolboxGPT", GPTTestDefinition)
    registry.add_test_definition("JaxToolboxGrok", GrokTestDefinition)
    registry.add_test_definition("JaxToolboxNemotron", NemotronTestDefinition)
    registry.add_test_definition("MegatronRun", MegatronRunTestDefinition)
    registry.add_test_definition("NcclTest", NCCLTestDefinition)
    registry.add_test_definition("NeMoLauncher", NeMoLauncherTestDefinition)
    registry.add_test_definition("NeMoRun", NeMoRunTestDefinition)
    registry.add_test_definition("NIXLBench", NIXLBenchTestDefinition)
    registry.add_test_definition("Sleep", SleepTestDefinition)
    registry.add_test_definition("SlurmContainer", SlurmContainerTestDefinition)
    registry.add_test_definition("TritonInference", TritonInferenceTestDefinition)
    registry.add_test_definition("UCCTest", UCCTestDefinition)

    # Register command generation strategies
    registry.add_strategy(
        CommandGenStrategy, [SlurmSystem], [ChakraReplayTestDefinition], ChakraReplaySlurmCommandGenStrategy
    )
    registry.add_strategy(
        CommandGenStrategy,
        [SlurmSystem],
        [JaxToolboxTestDefinition, GPTTestDefinition, GrokTestDefinition, NemotronTestDefinition],
        JaxToolboxSlurmCommandGenStrategy,
    )
    registry.add_strategy(
        CommandGenStrategy, [SlurmSystem], [MegatronRunTestDefinition], MegatronRunSlurmCommandGenStrategy
    )
    registry.add_strategy(CommandGenStrategy, [SlurmSystem], [NCCLTestDefinition], NcclTestSlurmCommandGenStrategy)
    registry.add_strategy(
        CommandGenStrategy, [SlurmSystem], [NeMoLauncherTestDefinition], NeMoLauncherSlurmCommandGenStrategy
    )
    registry.add_strategy(CommandGenStrategy, [SlurmSystem], [NeMoRunTestDefinition], NeMoRunSlurmCommandGenStrategy)
    registry.add_strategy(
        CommandGenStrategy, [SlurmSystem], [NIXLBenchTestDefinition], NIXLBenchSlurmCommandGenStrategy
    )
    registry.add_strategy(
        CommandGenStrategy, [StandaloneSystem], [SleepTestDefinition], SleepStandaloneCommandGenStrategy
    )
    registry.add_strategy(CommandGenStrategy, [LSFSystem], [SleepTestDefinition], SleepLSFCommandGenStrategy)
    registry.add_strategy(CommandGenStrategy, [SlurmSystem], [SleepTestDefinition], SleepSlurmCommandGenStrategy)
    registry.add_strategy(
        CommandGenStrategy, [SlurmSystem], [SlurmContainerTestDefinition], SlurmContainerCommandGenStrategy
    )
    registry.add_strategy(
        CommandGenStrategy, [SlurmSystem], [TritonInferenceTestDefinition], TritonInferenceSlurmCommandGenStrategy
    )
    registry.add_strategy(CommandGenStrategy, [SlurmSystem], [UCCTestDefinition], UCCTestSlurmCommandGenStrategy)

    # Register JSON generation strategies
    registry.add_strategy(JsonGenStrategy, [KubernetesSystem], [NCCLTestDefinition], NcclTestKubernetesJsonGenStrategy)
    registry.add_strategy(JsonGenStrategy, [RunAISystem], [NCCLTestDefinition], NcclTestRunAIJsonGenStrategy)
    registry.add_strategy(JsonGenStrategy, [KubernetesSystem], [SleepTestDefinition], SleepKubernetesJsonGenStrategy)

    # Register grading strategies
    registry.add_strategy(GradingStrategy, [SlurmSystem], [ChakraReplayTestDefinition], ChakraReplayGradingStrategy)
    registry.add_strategy(
        GradingStrategy,
        [SlurmSystem],
        [JaxToolboxTestDefinition, GPTTestDefinition, GrokTestDefinition, NemotronTestDefinition],
        JaxToolboxGradingStrategy,
    )
    registry.add_strategy(GradingStrategy, [SlurmSystem], [NCCLTestDefinition], NcclTestGradingStrategy)
    registry.add_strategy(GradingStrategy, [SlurmSystem], [NeMoLauncherTestDefinition], NeMoLauncherGradingStrategy)
    registry.add_strategy(GradingStrategy, [SlurmSystem], [SleepTestDefinition], SleepGradingStrategy)
    registry.add_strategy(GradingStrategy, [SlurmSystem], [UCCTestDefinition], UCCTestGradingStrategy)

    # Register job status retrieval strategies
    registry.add_strategy(
        JobStatusRetrievalStrategy, [SlurmSystem], [ChakraReplayTestDefinition], DefaultJobStatusRetrievalStrategy
    )
    registry.add_strategy(
        JobStatusRetrievalStrategy,
        [SlurmSystem],
        [JaxToolboxTestDefinition, GPTTestDefinition, GrokTestDefinition, NemotronTestDefinition],
        JaxToolboxJobStatusRetrievalStrategy,
    )
    registry.add_strategy(
        JobStatusRetrievalStrategy, [SlurmSystem], [MegatronRunTestDefinition], DefaultJobStatusRetrievalStrategy
    )
    registry.add_strategy(
        JobStatusRetrievalStrategy, [SlurmSystem], [NCCLTestDefinition], NcclTestJobStatusRetrievalStrategy
    )
    registry.add_strategy(
        JobStatusRetrievalStrategy, [KubernetesSystem], [NCCLTestDefinition], DefaultJobStatusRetrievalStrategy
    )
    registry.add_strategy(
        JobStatusRetrievalStrategy, [RunAISystem], [NCCLTestDefinition], DefaultJobStatusRetrievalStrategy
    )
    registry.add_strategy(
        JobStatusRetrievalStrategy, [SlurmSystem], [NeMoLauncherTestDefinition], NeMoLauncherSlurmJobIdRetrievalStrategy
    )
    registry.add_strategy(
        JobStatusRetrievalStrategy, [SlurmSystem], [NeMoRunTestDefinition], NeMoRunJobStatusRetrievalStrategy
    )
    registry.add_strategy(
        JobStatusRetrievalStrategy, [SlurmSystem], [NIXLBenchTestDefinition], NIXLBenchJobStatusRetrievalStrategy
    )
    registry.add_strategy(
        JobStatusRetrievalStrategy, [SlurmSystem], [SlurmContainerTestDefinition], DefaultJobStatusRetrievalStrategy
    )
    registry.add_strategy(
        JobStatusRetrievalStrategy, [SlurmSystem], [TritonInferenceTestDefinition], DefaultJobStatusRetrievalStrategy
    )
    registry.add_strategy(
        JobStatusRetrievalStrategy, [SlurmSystem], [UCCTestDefinition], DefaultJobStatusRetrievalStrategy
    )

    # Register report generation strategies
    registry.add_report(ChakraReplayTestDefinition, ChakraReplayReportGenerationStrategy)
    registry.add_report(GPTTestDefinition, JaxToolboxReportGenerationStrategy)
    registry.add_report(GrokTestDefinition, JaxToolboxReportGenerationStrategy)
    registry.add_report(JaxToolboxTestDefinition, JaxToolboxReportGenerationStrategy)
    registry.add_report(NemotronTestDefinition, JaxToolboxReportGenerationStrategy)
    registry.add_report(MegatronRunTestDefinition, CheckpointTimingReportGenerationStrategy)
    registry.add_report(NCCLTestDefinition, NcclTestPerformanceReportGenerationStrategy)
    registry.add_report(NeMoLauncherTestDefinition, NeMoLauncherReportGenerationStrategy)
    registry.add_report(NeMoRunTestDefinition, NeMoRunReportGenerationStrategy)
    registry.add_report(NeMoRunTestDefinition, NeMoRunDataStoreReportGenerationStrategy)
    registry.add_report(NIXLBenchTestDefinition, NIXLBenchReportGenerationStrategy)
    registry.add_report(SleepTestDefinition, SleepReportGenerationStrategy)
    registry.add_report(SlurmContainerTestDefinition, SlurmContainerReportGenerationStrategy)
    registry.add_report(TritonInferenceTestDefinition, TritonInferenceReportGenerationStrategy)
    registry.add_report(UCCTestDefinition, UCCTestReportGenerationStrategy)
