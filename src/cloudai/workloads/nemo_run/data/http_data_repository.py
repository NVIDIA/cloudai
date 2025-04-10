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


class HttpDataRepository:
    """Data repository using HTTP API (requests) instead of elastic client."""

    def __init__(
        self,
        post_endpoint: str,
        token: str,
        index: str,
        verify_certs: bool = True,
    ) -> None:
        self.post_endpoint = post_endpoint
        self.token = token
        self.index = index
        self.verify = verify_certs

    def store(self, entry: Dict[str, Any]) -> None:
        endpoint = f"{self.post_endpoint}/{self.index}/posting"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = json.dumps([entry])
        response = requests.post(endpoint, data=payload, headers=headers, verify=self.verify)
        response.raise_for_status()
