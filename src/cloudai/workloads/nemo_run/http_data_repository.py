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
from pathlib import Path
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

        config_path = Path(".cloudai.toml")
        if not config_path.is_file():
            raise ValueError(
                "Credential file '.cloudai.toml' not found. Please create the file and "
                "record a valid token under [data_repository]. Refer to USER_GUIDE.md for details."
            )

        credentials = toml.load(config_path)
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
