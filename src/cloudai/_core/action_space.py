# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Action-space primitives for CloudAI DSE.

CloudAI describes a tunable action space as a mapping from parameter name to
its candidate domain. Discrete parameters use a ``list`` of candidate values;
continuous parameters use :class:`ContinuousSpace`, a closed real interval.
Agents and adapters (e.g. ``GymnasiumAdapter``) read these to build their own
action representation and to decode sampled actions back to native values.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self


class ContinuousSpace(BaseModel):
    """
    A continuous (closed-interval) action-space dimension.

    Represents a single tunable parameter drawn from ``[low, high]``. ``dtype``
    declares whether decoded samples should be quantized to integers
    (``"int"``) or kept as floats (``"float"``); quantization is applied by
    consumers when decoding an action, not stored here.
    """

    model_config = ConfigDict(extra="forbid")

    low: float
    high: float
    dtype: Literal["int", "float"] = "float"

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        if self.low >= self.high:
            raise ValueError(f"ContinuousSpace requires low < high; got low={self.low}, high={self.high}")
        if self.dtype == "int" and (not self.low.is_integer() or not self.high.is_integer()):
            raise ValueError(
                f"ContinuousSpace(dtype='int') requires integer bounds; got low={self.low}, high={self.high}"
            )
        return self
