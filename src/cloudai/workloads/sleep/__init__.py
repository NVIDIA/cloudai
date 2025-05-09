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
)
from cloudai.registry import Registry
from cloudai.systems.kubernetes import KubernetesSystem
from cloudai.systems.lsf import LSFSystem
from cloudai.systems.slurm import SlurmSystem
from cloudai.systems.standalone import StandaloneSystem

from ..common import (
    DefaultJobStatusRetrievalStrategy,
    LSFJobIdRetrievalStrategy,
    SlurmJobIdRetrievalStrategy,
    StandaloneJobIdRetrievalStrategy,
)
from .grading_strategy import SleepGradingStrategy
from .kubernetes_json_gen_strategy import SleepKubernetesJsonGenStrategy
from .lsf_command_gen_strategy import SleepLSFCommandGenStrategy
from .report_generation_strategy import SleepReportGenerationStrategy
from .sleep import SleepCmdArgs, SleepTestDefinition
from .slurm_command_gen_strategy import SleepSlurmCommandGenStrategy
from .standalone_command_gen_strategy import SleepStandaloneCommandGenStrategy

Registry().add_strategy(
    CommandGenStrategy, [StandaloneSystem], [SleepTestDefinition], SleepStandaloneCommandGenStrategy
)
Registry().add_strategy(
    JobIdRetrievalStrategy, [StandaloneSystem], [SleepTestDefinition], StandaloneJobIdRetrievalStrategy
)
Registry().add_strategy(
    JobStatusRetrievalStrategy, [StandaloneSystem], [SleepTestDefinition], DefaultJobStatusRetrievalStrategy
)

Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [SleepTestDefinition], SleepSlurmCommandGenStrategy)
Registry().add_strategy(GradingStrategy, [SlurmSystem], [SleepTestDefinition], SleepGradingStrategy)
Registry().add_strategy(
    JobIdRetrievalStrategy,
    [SlurmSystem],
    [SleepTestDefinition],
    SlurmJobIdRetrievalStrategy,
)
Registry().add_strategy(
    JobStatusRetrievalStrategy, [SlurmSystem], [SleepTestDefinition], DefaultJobStatusRetrievalStrategy
)

Registry().add_strategy(JsonGenStrategy, [KubernetesSystem], [SleepTestDefinition], SleepKubernetesJsonGenStrategy)
Registry().add_strategy(
    JobStatusRetrievalStrategy, [KubernetesSystem], [SleepTestDefinition], DefaultJobStatusRetrievalStrategy
)

Registry().add_strategy(CommandGenStrategy, [LSFSystem], [SleepTestDefinition], SleepLSFCommandGenStrategy)
Registry().add_strategy(JobIdRetrievalStrategy, [LSFSystem], [SleepTestDefinition], LSFJobIdRetrievalStrategy)
Registry().add_strategy(
    JobStatusRetrievalStrategy, [LSFSystem], [SleepTestDefinition], DefaultJobStatusRetrievalStrategy
)

Registry().add_test_definition("Sleep", SleepTestDefinition)
Registry().add_report(SleepTestDefinition, SleepReportGenerationStrategy)

__all__ = [
    "SleepCmdArgs",
    "SleepGradingStrategy",
    "SleepKubernetesJsonGenStrategy",
    "SleepLSFCommandGenStrategy",
    "SleepReportGenerationStrategy",
    "SleepSlurmCommandGenStrategy",
    "SleepStandaloneCommandGenStrategy",
    "SleepTestDefinition",
]
