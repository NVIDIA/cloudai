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

import logging
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from cloudai.core import DockerImage, File, GitRepo, HFModel, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition

from .report_generation_strategy import CSV_FILES_PATTERN, JSON_FILES_PATTERN


class WorkerBaseArgs(BaseModel):
    """Base arguments for VLLM workers."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    num_nodes: int | list[int] = Field(
        default=1, serialization_alias="num-nodes", validation_alias=AliasChoices("num-nodes", "num_nodes")
    )
    nodes: str | None = Field(default=None)

    data_parallel_size: int | list[int] | None = Field(
        default=None,
        serialization_alias="data-parallel-size",
        validation_alias=AliasChoices("data-parallel-size", "data_parallel_size"),
    )
    gpu_memory_utilization: float | list[float] | None = Field(
        default=None,
        serialization_alias="gpu-memory-utilization",
        validation_alias=AliasChoices("gpu-memory-utilization", "gpu_memory_utilization"),
    )
    pipeline_parallel_size: int | list[int] | None = Field(
        default=None,
        serialization_alias="pipeline-parallel-size",
        validation_alias=AliasChoices("pipeline-parallel-size", "pipeline_parallel_size"),
    )
    tensor_parallel_size: int | list[int] | None = Field(
        default=None,
        serialization_alias="tensor-parallel-size",
        validation_alias=AliasChoices("tensor-parallel-size", "tensor_parallel_size"),
    )
    extra_args: str | list[str] | None = Field(
        default=None,
        serialization_alias="extra-args",
        validation_alias=AliasChoices("extra-args", "extra_args"),
    )


class PrefillWorkerArgs(WorkerBaseArgs):
    """Arguments for prefill worker."""

    pass


class DecodeWorkerArgs(WorkerBaseArgs):
    """Arguments for decode worker."""

    pass


class AIDynamoArgs(BaseModel):
    """Arguments for AI Dynamo setup."""

    model_config = ConfigDict(extra="allow")

    model: str = "Qwen/Qwen3-0.6B"
    backend: str = "vllm"
    workspace_path: str = Field(
        default="/workspace",
        serialization_alias="workspace-path",
        validation_alias=AliasChoices("workspace-path", "workspace_path"),
    )
    decode_worker: DecodeWorkerArgs = Field(default_factory=DecodeWorkerArgs)
    decode_cmd: str = Field(
        default="python3 -m dynamo.vllm",
        serialization_alias="decode-cmd",
        validation_alias=AliasChoices("decode-cmd", "decode_cmd"),
    )
    prefill_worker: PrefillWorkerArgs | None = None
    prefill_cmd: str = Field(
        default="python3 -m dynamo.vllm",
        serialization_alias="prefill-cmd",
        validation_alias=AliasChoices("prefill-cmd", "prefill_cmd"),
    )


class GenAIPerfArgs(BaseModel):
    """Arguments for GenAI performance profiling."""

    model_config = ConfigDict(extra="allow")

    extra_args: str | None = Field(
        default=None,
        serialization_alias="extra-args",
        validation_alias=AliasChoices("extra-args", "extra_args"),
    )


class AIDynamoCmdArgs(CmdArgs):
    """Arguments for AI Dynamo."""

    docker_image_url: str
    huggingface_home_container_path: Path = Path("/root/.cache/huggingface")
    dynamo: AIDynamoArgs
    genai_perf: GenAIPerfArgs
    run_script: str = ""


class AIDynamoTestDefinition(TestDefinition):
    """Test definition for AI Dynamo."""

    cmd_args: AIDynamoCmdArgs
    _docker_image: Optional[DockerImage] = None
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/ai_dynamo.sh")
    dynamo_repo: GitRepo = GitRepo(
        url="https://github.com/ai-dynamo/dynamo.git", commit="f7e468c7e8ff0d1426db987564e60572167e8464"
    )
    _hf_model: HFModel | None = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def hf_model(self) -> HFModel:
        if not self._hf_model:
            self._hf_model = HFModel(model_name=self.cmd_args.dynamo.model)
        return self._hf_model

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image, self.script, self.dynamo_repo, self.hf_model]

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        output_path = tr.output_path
        csv_files = list(output_path.rglob(CSV_FILES_PATTERN))
        json_files = list(output_path.rglob(JSON_FILES_PATTERN))
        logging.debug(f"Found CSV files in {output_path.absolute()}: {csv_files}, JSON files: {json_files}")
        has_results = len(csv_files) > 0 and len(json_files) > 0
        if not has_results:
            return JobStatusResult(False, "No result files found in the output directory.")
        return JobStatusResult(True)
