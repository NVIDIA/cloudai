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

from ..common import SlurmJobIdRetrievalStrategy
from .gpt import GPTCmdArgs, GPTTestDefinition
from .grading_strategy import JaxToolboxGradingStrategy
from .grok import GrokCmdArgs, GrokTestDefinition
from .jax_toolbox import JaxFdl, JaxToolboxCmdArgs, JaxToolboxTestDefinition
from .job_status_retrieval_strategy import JaxToolboxJobStatusRetrievalStrategy
from .nemotron import NemotronCmdArgs, NemotronTestDefinition
from .report_generation_strategy import JaxToolboxReportGenerationStrategy
from .slurm_command_gen_strategy import JaxToolboxSlurmCommandGenStrategy

Registry().add_test_definition("JaxToolboxGrok", GrokTestDefinition)
Registry().add_test_definition("JaxToolboxNemotron", NemotronTestDefinition)
Registry().add_test_definition("JaxToolboxGPT", GPTTestDefinition)

Registry().add_report(GPTTestDefinition, JaxToolboxReportGenerationStrategy)
Registry().add_report(GrokTestDefinition, JaxToolboxReportGenerationStrategy)
Registry().add_report(NemotronTestDefinition, JaxToolboxReportGenerationStrategy)


Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [GrokTestDefinition], JaxToolboxSlurmCommandGenStrategy)
Registry().add_strategy(GradingStrategy, [SlurmSystem], [GrokTestDefinition], JaxToolboxGradingStrategy)
Registry().add_strategy(JobIdRetrievalStrategy, [SlurmSystem], [GrokTestDefinition], SlurmJobIdRetrievalStrategy)
Registry().add_strategy(
    JobStatusRetrievalStrategy, [SlurmSystem], [GrokTestDefinition], JaxToolboxJobStatusRetrievalStrategy
)

Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [NemotronTestDefinition], JaxToolboxSlurmCommandGenStrategy)
Registry().add_strategy(GradingStrategy, [SlurmSystem], [NemotronTestDefinition], JaxToolboxGradingStrategy)
Registry().add_strategy(JobIdRetrievalStrategy, [SlurmSystem], [NemotronTestDefinition], SlurmJobIdRetrievalStrategy)
Registry().add_strategy(
    JobStatusRetrievalStrategy, [SlurmSystem], [NemotronTestDefinition], JaxToolboxJobStatusRetrievalStrategy
)

Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [GPTTestDefinition], JaxToolboxSlurmCommandGenStrategy)
Registry().add_strategy(GradingStrategy, [SlurmSystem], [GPTTestDefinition], JaxToolboxGradingStrategy)
Registry().add_strategy(JobIdRetrievalStrategy, [SlurmSystem], [GPTTestDefinition], SlurmJobIdRetrievalStrategy)
Registry().add_strategy(
    JobStatusRetrievalStrategy, [SlurmSystem], [GPTTestDefinition], JaxToolboxJobStatusRetrievalStrategy
)

__all__ = [
    "GPTCmdArgs",
    "GPTTestDefinition",
    "GrokCmdArgs",
    "GrokTestDefinition",
    "JaxFdl",
    "JaxToolboxCmdArgs",
    "JaxToolboxGradingStrategy",
    "JaxToolboxJobStatusRetrievalStrategy",
    "JaxToolboxReportGenerationStrategy",
    "JaxToolboxSlurmCommandGenStrategy",
    "JaxToolboxTestDefinition",
    "NemotronCmdArgs",
    "NemotronTestDefinition",
]
