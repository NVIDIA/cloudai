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

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, model_validator

from cloudai.core import CmdArgs, Installable, TestDefinition


class Agg(BaseModel):
    """Aggregated (IFB) configuration."""

    model_config = ConfigDict(extra="allow")

    batch_size: Union[int, List[int]]
    ctx_tokens: Union[int, List[int]]
    tp: Union[int, List[int]] = 1
    pp: Union[int, List[int]] = 1
    dp: Union[int, List[int]] = 1
    moe_tp: Union[int, List[int]] = 1
    moe_ep: Union[int, List[int]] = 1


class Disagg(BaseModel):
    """Disaggregated configuration."""

    model_config = ConfigDict(extra="allow")

    p_tp: Union[int, List[int]]
    p_pp: Union[int, List[int]]
    p_dp: Union[int, List[int]]
    p_bs: Union[int, List[int]]
    p_workers: Union[int, List[int]]

    d_tp: Union[int, List[int]]
    d_pp: Union[int, List[int]]
    d_dp: Union[int, List[int]]
    d_bs: Union[int, List[int]]
    d_workers: Union[int, List[int]]

    prefill_correction_scale: float = 1.0
    decode_correction_scale: float = 1.0


class AiconfiguratorCmdArgs(CmdArgs):
    """Command arguments for Aiconfigurator workload with nested agg/disagg configs."""

    python_executable: str = "python"

    model_name: str
    system: str
    backend: str = "trtllm"
    version: str = "0.20.0"

    isl: Union[int, List[int]]
    osl: Union[int, List[int]]

    agg: Optional[Agg] = None
    disagg: Optional[Disagg] = None

    @model_validator(mode="after")
    def _validate_agg_disagg(self) -> "AiconfiguratorCmdArgs":
        if self.agg is not None and self.disagg is not None:
            raise ValueError("Only one of 'agg' or 'disagg' may be specified.")
        if self.agg is None and self.disagg is None:
            raise ValueError("Either 'agg' or 'disagg' must be specified.")
        return self


class AiconfiguratorTestDefinition(TestDefinition):
    """Test object for running Aiconfigurator predictor as a workload."""

    cmd_args: AiconfiguratorCmdArgs

    @property
    def installables(self) -> list[Installable]:
        return []
