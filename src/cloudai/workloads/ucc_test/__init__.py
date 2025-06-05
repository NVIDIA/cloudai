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

from .grading_strategy import UCCTestGradingStrategy
from .report_generation_strategy import UCCTestReportGenerationStrategy
from .slurm_command_gen_strategy import UCCTestSlurmCommandGenStrategy
from .ucc import UCCCmdArgs, UCCTestDefinition

__all__ = [
    "UCCCmdArgs",
    "UCCTestDefinition",
    "UCCTestGradingStrategy",
    "UCCTestReportGenerationStrategy",
    "UCCTestSlurmCommandGenStrategy",
]
