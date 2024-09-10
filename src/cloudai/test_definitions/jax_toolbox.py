# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai import CmdArgs, TestDefinition
from pydantic import BaseModel, ConfigDict


class JaxFdl(BaseModel):
    """JAX FDL configuration."""

    model_config = ConfigDict(extra="forbid")
    num_gpus: int = 64
    num_groups: int = 64
    max_steps: int = 20
    num_layers: int = 1
    num_stages: int = 1
    num_microbatches: int = 1
    use_repeated_layer: bool = False
    percore_batch_size: int = 1
    checkpoint_policy: str = "save_nothing"
    ici_mesh_shape: str = "'[1, 1, 8, 1]'"
    dcn_mesh_shape: str = "'[1, 8, 1, 1]'"


class JaxToolboxCmdArgs(CmdArgs):
    """JAX Toolbox test command arguments."""

    pass


class JaxToolboxTestDefinition(TestDefinition):
    """Test object for JAX Toolbox."""

    pass
