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

from pydantic import BaseModel

from .jax_toolbox import JaxFdl, JaxToolboxCmdArgs, JaxToolboxTestDefinition, NCCLPreTest


class XLAFlags(BaseModel):
    """XLA flags configuration."""

    xla_gpu_run_post_layout_collective_pipeliner: bool = False


class GrokFdl(JaxFdl):
    """Fool data loader for Grok."""

    use_fp8: bool = False
    use_te_dpa: bool = False
    checkpoint_policy: str = '"save_iteration_input"'


class GrokCmdArgs(JaxToolboxCmdArgs):
    """Grok test command arguments."""

    fdl: GrokFdl
    fdl_config: Optional[str] = None
    enable_pgle: bool = False
    pre_test: Optional[NCCLPreTest] = None
    profile: Optional[XLAFlags] = None
    perf: Optional[XLAFlags] = None


class GrokTestDefinition(JaxToolboxTestDefinition):
    """Test object for Grok."""

    cmd_args: GrokCmdArgs
