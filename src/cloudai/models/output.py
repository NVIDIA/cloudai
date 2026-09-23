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

import datetime
from typing import Literal

import pydantic

Status = Literal["pending", "running", "completed", "failed", "cancelled", "unknown"]


class Dimension(pydantic.BaseModel):
    """One coordinate of a metric measurement."""

    name: str
    value: str
    unit: str = ""
    is_x: bool = False


class Metric(pydantic.BaseModel):
    """A canonical measurement at a dimension point."""

    name: str
    value: str | int | pydantic.FiniteFloat | bool
    unit: str = ""
    dimensions: list[Dimension] = pydantic.Field(default_factory=list)


class Run(pydantic.BaseModel):
    """A logical execution with normalized metadata and original measurements."""

    path: str
    jobid: str
    status: Status = "unknown"
    metrics: list[Metric] = pydantic.Field(default_factory=list)
    start: datetime.datetime | None = None
    finish: datetime.datetime | None = None
    duration: pydantic.FiniteFloat | None = None
    iteration: int | None = None
    step: int | None = None


class DSE(pydantic.BaseModel):
    """Search space and recommendation for one test case."""

    space: dict[str, list[str | int | pydantic.FiniteFloat]]
    best_config: dict[str, str | int | pydantic.FiniteFloat] | None = None
    best_step: int | None = None


class Test(pydantic.BaseModel):
    """A test case with all logical executions and optional DSE metadata."""

    id: str
    name: str
    description: str | None = None
    status: Status = "pending"
    path: str
    metrics: list[Metric] = pydantic.Field(default_factory=list)
    runs: list[Run] = pydantic.Field(default_factory=list)
    dse: DSE | None = None


class Experiment(pydantic.BaseModel):
    """Full snapshot spanning the entire scenario, including all DSE trials."""

    id: str
    name: str
    description: str | None = None
    status: Status = "pending"
    path: str
    start: datetime.datetime | None = None
    finish: datetime.datetime | None = None
    duration: pydantic.FiniteFloat | None = None
    tests: list[Test] = pydantic.Field(default_factory=list)
