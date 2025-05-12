# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from .report_generation_strategy import TritonInferenceReportGenerationStrategy
from .slurm_command_gen_strategy import TritonInferenceSlurmCommandGenStrategy
from .triton_inference import TritonInferenceCmdArgs, TritonInferenceTestDefinition

Registry().add_strategy(
    CommandGenStrategy, [SlurmSystem], [TritonInferenceTestDefinition], TritonInferenceSlurmCommandGenStrategy
)
Registry().add_strategy(
    JobIdRetrievalStrategy, [SlurmSystem], [TritonInferenceTestDefinition], SlurmJobIdRetrievalStrategy
)
Registry().add_strategy(
    JobStatusRetrievalStrategy, [SlurmSystem], [TritonInferenceTestDefinition], DefaultJobStatusRetrievalStrategy
)

Registry().add_test_definition("TritonInference", TritonInferenceTestDefinition)
Registry().add_report(TritonInferenceTestDefinition, TritonInferenceReportGenerationStrategy)

__all__ = [
    "TritonInferenceCmdArgs",
    "TritonInferenceReportGenerationStrategy",
    "TritonInferenceSlurmCommandGenStrategy",
    "TritonInferenceTestDefinition",
]
