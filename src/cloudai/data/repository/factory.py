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
from typing import Any, Dict

from .base import BaseDataRepository
from .elastic import ElasticDataRepository
from .http import HttpDataRepository


def create_data_repository(config: Dict[str, Any]) -> BaseDataRepository:
    backend = config.get("backend")
    if not backend:
        raise ValueError("Missing 'backend' configuration for data repository.")

    if backend == "http":
        try:
            return HttpDataRepository(
                token=config["token"],
                index=config["index"],
                base_url_data=config["base_url_data"],
                base_url_search=config["base_url_search"],
                verify_certs=config.get("verify_certs", True),
            )
        except KeyError as e:
            raise ValueError(f"Missing required HTTP config key: {e}") from e

    elif backend == "elastic":
        try:
            api_key = (config["api_key_id"], config["api_key_secret"])
            return ElasticDataRepository(
                host=config["host"],
                api_key=api_key,
                index=config["index"],
                verify_certs=config.get("verify_certs", True),
            )
        except KeyError as e:
            raise ValueError(f"Missing required Elastic config key: {e}") from e

    else:
        raise ValueError(f"Unsupported backend '{backend}' for data repository.")
