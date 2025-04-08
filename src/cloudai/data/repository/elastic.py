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
from typing import Any, Dict, Tuple

from elasticsearch import Elasticsearch, NotFoundError

from .base import BaseDataRepository


class ElasticDataRepository(BaseDataRepository):
    """A repository for storing and retrieving data using Elasticsearch."""

    def __init__(self, host: str, api_key: Tuple[str, str], index: str, verify_certs: bool = True) -> None:
        self.index = index
        self.client = Elasticsearch([host], api_key=api_key, verify_certs=verify_certs)

    def store(self, entry: Dict[str, Any]) -> None:
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
        self.client.index(index=self.index, body=entry)

    def retrieve(self, identifier: str) -> Dict[str, Any]:
        try:
            resp = self.client.get(index=self.index, id=identifier)
            return resp.get("_source", {})
        except NotFoundError:
            return {}

    def search(self, query: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.search(index=self.index, body=query)
