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


class SystemContext(BaseModel):
    """
    Hold system metadata common to all workloads.

    Attributes:
      user: The user associated with the system.
      cluster: The cluster identifier for the system.
      num_gpus: The number of GPUs associated with the system.
    """

    user: Optional[str] = Field(default=None, alias="s_user")
    cluster: Optional[str] = Field(default=None, alias="s_cluster")
    num_gpus: Optional[int] = Field(default=None, alias="l_num_gpus")

    model_config = {
        "populate_by_name": True,
    }
