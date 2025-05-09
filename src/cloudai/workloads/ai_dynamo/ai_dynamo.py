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
from typing import List, Literal, Optional

from cloudai import CmdArgs, DockerImage, Installable, TestDefinition


class AIDynamoCmdArgs(CmdArgs):
    """Arguments for AI Dynamo."""

    docker_image_url: str
    served_model_name: str
    num_prefill_nodes: int
    num_decode_nodes: int
    port_etcd: int = 2379
    port_nats: int = 4222
    config_path: str
    port: int = 8000
    endpoint_type: Literal["chat"] = "chat"
    service_kind: Literal["openai"] = "openai"
    endpoint: str = "v1/chat/completions"
    streaming: bool
    warmup_request_count: int
    random_seed: int
    synthetic_input_tokens_mean: int
    synthetic_input_tokens_stddev: int
    output_tokens_mean: int
    output_tokens_stddev: int
    extra_inputs: Optional[str]
    concurrency: int
    request_count: int
    sleep_seconds: int = 550


class AIDynamoTestDefinition(TestDefinition):
    """Test definition for AI Dynamo."""

    cmd_args: AIDynamoCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> List[Installable]:
        return [self.docker_image]

    @property
    def hugging_face_home_path(self) -> Path:
        raw = self.extra_env_vars.get("HF_HOME")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("HF_HOME must be set and non-empty")
        path = Path(raw)
        if not path.is_dir():
            raise FileNotFoundError(f"HF_HOME path not found at {path}")
        return path
