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

from typing import Optional

from pydantic import Field

from cloudai import CmdArgs, TestDefinition


class ChakraReplayCmdArgs(CmdArgs):
    """ChakraReplay test command arguments."""

    docker_image_url: str = Field(default="DOCKER_IMAGE_URL")
    mpi: str = Field(default="pmix")
    trace_type: str = Field(default="et")
    trace_path: Optional[str] = None
    backend: str = Field(default="nccl")
    device: str = Field(default="cuda")


class ChakraReplayTestDefinition(TestDefinition):
    """Test object for ChakraReplay."""

    cmd_args: ChakraReplayCmdArgs
