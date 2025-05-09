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

from cloudai import CommandGenStrategy, GradingStrategy, JobIdRetrievalStrategy, JobStatusRetrievalStrategy
from cloudai.registry import Registry
from cloudai.systems.slurm import SlurmSystem

from ..common import DefaultJobStatusRetrievalStrategy, SlurmJobIdRetrievalStrategy
from .grading_strategy import UCCTestGradingStrategy
from .report_generation_strategy import UCCTestReportGenerationStrategy
from .slurm_command_gen_strategy import UCCTestSlurmCommandGenStrategy
from .ucc import UCCCmdArgs, UCCTestDefinition

Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [UCCTestDefinition], UCCTestSlurmCommandGenStrategy)
Registry().add_strategy(GradingStrategy, [SlurmSystem], [UCCTestDefinition], UCCTestGradingStrategy)
Registry().add_strategy(JobIdRetrievalStrategy, [SlurmSystem], [UCCTestDefinition], SlurmJobIdRetrievalStrategy)
Registry().add_strategy(
    JobStatusRetrievalStrategy, [SlurmSystem], [UCCTestDefinition], DefaultJobStatusRetrievalStrategy
)

Registry().add_test_definition("UCCTest", UCCTestDefinition)
Registry().add_report(UCCTestDefinition, UCCTestReportGenerationStrategy)

__all__ = [
    "UCCCmdArgs",
    "UCCTestDefinition",
    "UCCTestGradingStrategy",
    "UCCTestReportGenerationStrategy",
    "UCCTestSlurmCommandGenStrategy",
]
