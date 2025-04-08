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

from pydantic import BaseModel, Field


class JobContext(BaseModel):
    """
    Hold job metadata common to all workloads.

    Attributes:
      job_id: The job id provided by the service provider.
      image: The path or identifier for the software image.
      provider: The service provider (e.g., 'eos', 'aws-iad-cs-002').
      mode: The job mode, for example "training" or "inference".
      gpus_per_node: GPUs available per node (numeric configuration).
      num_nodes: Number of nodes used in the job.
      profiling: An optional flag whether profiling is enabled.
    """

    job_id: Optional[str] = Field(default=None, alias="s_job_id")
    image: Optional[str] = Field(default=None, alias="s_image")
    num_nodes: Optional[int] = Field(default=None, ge=1, alias="l_num_nodes")
    profiling: Optional[bool] = None

    model_config = {
        "populate_by_name": True,
    }
