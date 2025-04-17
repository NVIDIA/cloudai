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
import toml


class HttpDataRepository:
    """Data repository using HTTP API (requests) instead of elastic client."""

    def __init__(
        self,
        endpoint: str,
        verify_certs: bool = True,
    ) -> None:
        self.endpoint = endpoint
        self.verify = verify_certs

        credentials = toml.load(".cloudai.toml")
        self.token = credentials.get("data_repository", {}).get("token")
        if not self.token:
            raise ValueError(
                "Credential not configured. Please create and populate the .cloudai.toml file "
                "with the token under the data_repository section."
            )

    def push(self, entry: Dict[str, Any]) -> None:
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = json.dumps([entry])
        response = requests.post(self.endpoint, data=payload, headers=headers, verify=self.verify)
        response.raise_for_status()
