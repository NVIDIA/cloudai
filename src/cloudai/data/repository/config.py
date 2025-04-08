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

from pydantic import BaseModel

from .base import BaseDataRepository
from .factory import create_data_repository


class DataRepositoryConfig(BaseModel):
    """Configuration for a data repository."""

    backend: str
    post_endpoint: Optional[str] = None
    search_endpoint: Optional[str] = None
    token: Optional[str] = None
    host: Optional[str] = None
    api_key_id: Optional[str] = None
    api_key_secret: Optional[str] = None
    index: Optional[str] = None
    verify_certs: bool = True

    def instantiate_repository(self) -> BaseDataRepository:
        return create_data_repository(self.dict())
