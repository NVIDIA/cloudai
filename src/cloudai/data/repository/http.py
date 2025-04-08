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

import json
from typing import Any, Dict

import requests

from .base import BaseDataRepository


class HttpDataRepository(BaseDataRepository):
    """Data repository using HTTP API (requests) instead of elastic client."""

    def __init__(
        self,
        token: str,
        index: str,
        base_url_data: str,
        base_url_search: str,
        verify_certs: bool = True,
    ) -> None:
        self.token = token
        self.index = index
        self.base_url_data = base_url_data
        self.base_url_search = base_url_search
        self.verify = verify_certs

    def store(self, entry: Dict[str, Any]) -> None:
        endpoint = f"{self.base_url_data}/{self.index}/posting"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = json.dumps([entry])
        response = requests.post(endpoint, data=payload, headers=headers, verify=self.verify)
        response.raise_for_status()

    def retrieve(self, identifier: str) -> Dict[str, Any]:
        query = {"query": {"match": {"_id": identifier}}}
        results = self.search(query)
        return results.get("hits", {}).get("hits", [{}])[0].get("_source", {})

    def search(self, query: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = f"{self.base_url_search}/res/v1/es/{self.index}/_search"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        response = requests.post(endpoint, headers=headers, data=json.dumps(query), verify=self.verify)
        response.raise_for_status()
        return response.json()
