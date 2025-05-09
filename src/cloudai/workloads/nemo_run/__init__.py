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

from cloudai import CommandGenStrategy, JobIdRetrievalStrategy, JobStatusRetrievalStrategy
from cloudai.registry import Registry
from cloudai.systems.slurm import SlurmSystem

from ..common import DefaultJobStatusRetrievalStrategy, SlurmJobIdRetrievalStrategy
from .data_store_report_generation_strategy import NeMoRunDataStoreReportGenerationStrategy
from .nemo_run import Data, Log, LogCkpt, NeMoRunCmdArgs, NeMoRunTestDefinition, Trainer, TrainerStrategy
from .report_generation_strategy import NeMoRunReportGenerationStrategy
from .slurm_command_gen_strategy import NeMoRunSlurmCommandGenStrategy

Registry().add_strategy(CommandGenStrategy, [SlurmSystem], [NeMoRunTestDefinition], NeMoRunSlurmCommandGenStrategy)
Registry().add_strategy(JobIdRetrievalStrategy, [SlurmSystem], [NeMoRunTestDefinition], SlurmJobIdRetrievalStrategy)
Registry().add_strategy(
    JobStatusRetrievalStrategy, [SlurmSystem], [NeMoRunTestDefinition], DefaultJobStatusRetrievalStrategy
)

Registry().add_test_definition("NeMoRun", NeMoRunTestDefinition)
Registry().add_report(NeMoRunTestDefinition, NeMoRunReportGenerationStrategy)
Registry().add_report(NeMoRunTestDefinition, NeMoRunDataStoreReportGenerationStrategy)

__all__ = [
    "Data",
    "Log",
    "LogCkpt",
    "NeMoRunCmdArgs",
    "NeMoRunDataStoreReportGenerationStrategy",
    "NeMoRunReportGenerationStrategy",
    "NeMoRunSlurmCommandGenStrategy",
    "NeMoRunTestDefinition",
    "Trainer",
    "TrainerStrategy",
]
