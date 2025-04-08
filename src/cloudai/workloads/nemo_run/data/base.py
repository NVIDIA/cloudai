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

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from cloudai.data.model.base_record import BaseRecord


class PerformanceMetrics(BaseModel):
    """Performance metrics for NemoRun workloads."""

    metric: float = Field(alias="d_metric")
    metric_stddev: float = Field(alias="d_metric_stddev")
    step_time_mean: float = Field(alias="d_step_time_mean")
    tokens_per_sec: float = Field(alias="d_tokens_per_sec")
    checkpoint_size: Optional[int] = Field(default=None, alias="l_checkpoint_size")
    checkpoint_save_rank_time: Optional[float] = Field(alias="d_checkpoint_save_rank_time")

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class NemoRunBaseRecord(BaseRecord):
    """NemoRun base record for common top-level fields."""

    mode: str = Field(default="training", alias="s_job_mode")
    synthetic_dataset: bool = Field(alias="b_synthetic_dataset")
    performance: PerformanceMetrics
