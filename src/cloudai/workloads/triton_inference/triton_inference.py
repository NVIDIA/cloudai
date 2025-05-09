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

from pathlib import Path
from typing import Literal, Optional

from cloudai import CmdArgs, DockerImage, Installable, TestDefinition


class TritonInferenceCmdArgs(CmdArgs):
    """Arguments for TritonInference server and client."""

    server_docker_image_url: str
    client_docker_image_url: str
    served_model_name: str
    endpoint_type: Literal["chat"] = "chat"
    service_kind: Literal["openai"] = "openai"
    streaming: bool = True
    port: int = 8000
    num_prompts: int = 20
    input_sequence_length: int = 128
    output_sequence_length: int = 128
    concurrency: int = 1
    tokenizer: str
    sleep_seconds: int = 3300


class TritonInferenceTestDefinition(TestDefinition):
    """Test definition for TritonInference server and NIM client."""

    cmd_args: TritonInferenceCmdArgs
    _server_image: Optional[DockerImage] = None
    _client_image: Optional[DockerImage] = None

    @property
    def server_docker_image(self) -> DockerImage:
        if not self._server_image:
            self._server_image = DockerImage(url=self.cmd_args.server_docker_image_url)
        return self._server_image

    @property
    def client_docker_image(self) -> DockerImage:
        if not self._client_image:
            self._client_image = DockerImage(url=self.cmd_args.client_docker_image_url)
        return self._client_image

    @property
    def installables(self) -> list[Installable]:
        return [self.server_docker_image, self.client_docker_image]

    @property
    def nim_model_path(self) -> Path:
        raw = self.extra_env_vars.get("NIM_MODEL_NAME")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("NIM_MODEL_NAME must be set and non-empty")
        path = Path(raw)
        if not path.is_dir():
            raise FileNotFoundError(f"NIM_MODEL_NAME path not found at {path}")
        return path

    @property
    def nim_cache_path(self) -> Path:
        raw = self.extra_env_vars.get("NIM_CACHE_PATH")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("NIM_CACHE_PATH must be set and non-empty")
        path = Path(raw)
        if not path.is_dir():
            raise FileNotFoundError(f"NIM_CACHE_PATH path not found at {path}")
        return path
