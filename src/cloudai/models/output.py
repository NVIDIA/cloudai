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

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, FiniteFloat

Status = Literal["pending", "running", "completed", "failed", "cancelled", "unknown"]


class Dimension(BaseModel):
    """One coordinate of a metric measurement."""

    name: str
    value: str
    unit: str = ""
    is_x: bool = False


class Metric(BaseModel):
    """A canonical measurement at a dimension point."""

    name: str
    value: str | int | FiniteFloat | bool
    unit: str = ""
    dimensions: list[Dimension] = Field(default_factory=list)


class Run(BaseModel):
    """A logical execution with normalized metadata and original measurements."""

    path: str
    jobid: str
    status: Status = "unknown"
    metrics: list[Metric] = Field(default_factory=list)
    start: datetime | None = None
    finish: datetime | None = None
    duration: FiniteFloat | None = None
    iteration: int | None = None
    step: int | None = None


class DSE(BaseModel):
    """Search space and recommendation for one test case."""

    space: dict[str, list[str | int | FiniteFloat]]
    best_config: dict[str, str | int | FiniteFloat] | None = None
    best_step: int | None = None


class TestShort(BaseModel):
    """Test identity, status, and summary metrics."""

    id: str
    name: str
    description: str | None = None
    status: Status = "pending"
    path: str
    metrics: list[Metric] = Field(default_factory=list)


class Test(TestShort):
    """A test case with all logical executions and optional DSE metadata."""

    runs: list[Run] = Field(default_factory=list)
    dse: DSE | None = None


class _ExperimentMetadata(BaseModel):
    id: str
    name: str
    description: str | None = None
    status: Status = "pending"
    path: str
    start: datetime | None = None
    finish: datetime | None = None
    duration: FiniteFloat | None = None


class ExperimentShort(_ExperimentMetadata):
    """Catalog snapshot without individual runs or DSE details."""

    tests: list[TestShort] = Field(default_factory=list)


class Experiment(_ExperimentMetadata):
    """Full snapshot spanning the entire scenario, including all DSE trials."""

    tests: list[Test] = Field(default_factory=list)
