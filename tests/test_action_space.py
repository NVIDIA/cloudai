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

import pytest
from pydantic import ValidationError

from cloudai._core.action_space import ContinuousSpace


def test_continuous_space_defaults_to_float() -> None:
    space = ContinuousSpace(low=0.0, high=1.0)
    assert space.low == 0.0
    assert space.high == 1.0
    assert space.dtype == "float"


def test_continuous_space_coerces_int_bounds_and_keeps_int_dtype() -> None:
    space = ContinuousSpace(low=0, high=200, dtype="int")
    assert space.dtype == "int"
    assert isinstance(space.low, float) and isinstance(space.high, float)


def test_continuous_space_rejects_low_ge_high() -> None:
    with pytest.raises(ValidationError, match="low < high"):
        ContinuousSpace(low=1.0, high=1.0)
    with pytest.raises(ValidationError, match="low < high"):
        ContinuousSpace(low=2.0, high=1.0)


def test_continuous_space_rejects_unknown_dtype() -> None:
    with pytest.raises(ValidationError):
        ContinuousSpace(low=0.0, high=1.0, dtype="double")  # type: ignore


def test_continuous_space_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ContinuousSpace(low=0.0, high=1.0, step=0.1)  # type: ignore
