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

from pydantic import BaseModel

from cloudai import CmdArgs, DockerImage, Installable, TestDefinition


class FrontendArgs(BaseModel):
    """Arguments for the frontend role in AI Dynamo."""

    port_etcd: int = 2379
    port_nats: int = 4222


class PrefillArgs(BaseModel):
    """Arguments for the prefill role in AI Dynamo."""

    num_nodes: int


class DecodeArgs(BaseModel):
    """Arguments for the decode role in AI Dynamo."""

    num_nodes: int


class AIDynamoArgs(BaseModel):
    """Arguments for AI Dynamo setup."""

    config_path: str
    frontend: FrontendArgs
    prefill: PrefillArgs
    decode: DecodeArgs


class GenAIPerfArgs(BaseModel):
    """Arguments for GenAI performance profiling."""

    port: int = 8000
    served_model_name: str
    endpoint: Optional[str] = None
    endpoint_type: str = "kserve"
    service_kind: Literal["openai"] = "openai"
    streaming: bool
    extra_inputs: Optional[str]
    input_file: Optional[str] = None
    output_tokens_mean: int
    osl: int = -1
    output_tokens_stddev: int = 0
    random_seed: int
    request_count: int
    synthetic_input_tokens_mean: int
    isl: int = 550
    synthetic_input_tokens_stddev: int = 0
    warmup_request_count: int
    concurrency: Optional[int] = None
    request_rate: Optional[float] = None


class AIDynamoCmdArgs(CmdArgs):
    """Arguments for AI Dynamo."""

    docker_image_url: str
    dynamo: AIDynamoArgs
    sleep_seconds: int = 550
    genai_perf: GenAIPerfArgs


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
