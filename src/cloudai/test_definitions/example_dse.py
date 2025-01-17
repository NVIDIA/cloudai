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

from typing import Union

from cloudai import CmdArgs, TestDefinition
from cloudai.installer.installables import Installable


class ExampleDSECmdArgs(CmdArgs):
    """Command line arguments for the Example DSE test."""

    num_cores: Union[int, list[int]]
    freq: Union[int, list[int]]
    mem_type: Union[int, list[int]]
    mem_size: Union[int, list[int]]


class ExampleDSETestDefinition(TestDefinition):
    """Test definition for Example DSE."""

    cmd_args: ExampleDSECmdArgs

    @property
    def installables(self) -> list[Installable]:
        return []
