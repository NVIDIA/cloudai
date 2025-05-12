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

from .chakra_replay import ChakraReplayTestDefinition
from .jax_toolbox import GPTTestDefinition, GrokTestDefinition, NemotronTestDefinition
from .megatron_run import MegatronRunTestDefinition
from .nemo_launcher import NeMoLauncherTestDefinition
from .nemo_run import NeMoRunTestDefinition
from .sleep import SleepTestDefinition
from .slurm_container import SlurmContainerTestDefinition
from .triton_inference import TritonInferenceTestDefinition
from .ucc_test import UCCTestDefinition

__all__ = [
    "ChakraReplayTestDefinition",
    "GPTTestDefinition",
    "GrokTestDefinition",
    "MegatronRunTestDefinition",
    "NeMoLauncherTestDefinition",
    "NeMoRunTestDefinition",
    "NemotronTestDefinition",
    "SleepTestDefinition",
    "SlurmContainerTestDefinition",
    "TritonInferenceTestDefinition",
    "UCCTestDefinition",
]
