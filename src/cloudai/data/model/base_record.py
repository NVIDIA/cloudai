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

from datetime import datetime

from pydantic import BaseModel, Field

from .gsw_context import GSWContext


class BaseRecord(BaseModel):
    """Base record for common fields across records."""

    timestamp: datetime = Field(alias="ts_timestamp")
    created_ts: int = Field(alias="ts_created")
    gsw: GSWContext

    model_config = {
        "populate_by_name": True,
    }

    def to_flat_dict(self) -> dict:
        """Flatten the record into a dictionary using alias names."""
        return self.dict(by_alias=True)
