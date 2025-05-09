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

from cloudai.registry import Registry

from .metadata import SlurmSystemMetadata
from .slurm_command_gen_strategy import SlurmCommandGenStrategy
from .slurm_installer import SlurmInstaller
from .slurm_node import SlurmNode, SlurmNodeState
from .slurm_runner import SlurmRunner
from .slurm_system import SlurmGroup, SlurmPartition, SlurmSystem, parse_node_list

Registry().add_installer("slurm", SlurmInstaller)
Registry().add_system("slurm", SlurmSystem)
Registry().add_runner("slurm", SlurmRunner)

__all__ = [
    "SlurmCommandGenStrategy",
    "SlurmGroup",
    "SlurmInstaller",
    "SlurmNode",
    "SlurmNodeState",
    "SlurmPartition",
    "SlurmRunner",
    "SlurmSystem",
    "SlurmSystemMetadata",
    "parse_node_list",
]
