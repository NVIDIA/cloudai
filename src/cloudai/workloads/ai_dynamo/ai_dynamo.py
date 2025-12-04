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
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from cloudai.core import DockerImage, File, GitRepo, HFModel, Installable, JobStatusResult, PythonExecutable, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition

from .report_generation_strategy import CSV_FILES_PATTERN, JSON_FILES_PATTERN


class WorkerBaseArgs(BaseModel):
    """Base arguments for VLLM workers."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    num_nodes: Union[int, list[int]] = Field(alias="num-nodes")
    nodes: Optional[str] = Field(default=None, alias="nodes")


class PrefillWorkerArgs(WorkerBaseArgs):
    """Arguments for prefill worker."""

    pass


class DecodeWorkerArgs(WorkerBaseArgs):
    """Arguments for decode worker."""

    pass


class AIDynamoArgs(BaseModel):
    """Arguments for AI Dynamo setup."""

    model_config = ConfigDict(extra="allow")

    model: str
    backend: str = "vllm"
    prefill_worker: PrefillWorkerArgs
    decode_worker: DecodeWorkerArgs


class GenAIPerfArgs(BaseModel):
    """Arguments for GenAI performance profiling."""

    model_config = ConfigDict(extra="allow")


class AIDynamoCmdArgs(CmdArgs):
    """Arguments for AI Dynamo."""

    docker_image_url: str
    huggingface_home_container_path: Path = Path("/root/.cache/huggingface")
    dynamo: AIDynamoArgs
    genai_perf: GenAIPerfArgs
    run_script: str = ""
    dynamo_graph_path: Optional[str] = None

    def model_post_init(self, *args, **kwargs) -> None:
        """Post-init validation of fields."""
        super().model_post_init(*args, **kwargs)
        if self.dynamo_graph_path is not None and not Path(self.dynamo_graph_path).exists():
            raise ValueError(f"Dynamo graph file not found at {self.dynamo_graph_path}")


class AIDynamoTestDefinition(TestDefinition):
    """Test definition for AI Dynamo."""

    cmd_args: AIDynamoCmdArgs
    _docker_image: Optional[DockerImage] = None
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/ai_dynamo.sh")
    dynamo_repo: GitRepo = GitRepo(
        url="https://github.com/ai-dynamo/dynamo.git", commit="f7e468c7e8ff0d1426db987564e60572167e8464"
    )
    genai_perf_repo: GitRepo = GitRepo(
        url="https://github.com/triton-inference-server/perf_analyzer.git",
        commit="3c0bc9efa1844a82dfcc911f094f5026e6dd9214",
    )
    _python_executable: Optional[PythonExecutable] = None
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
        return [self.docker_image, self.script, self.dynamo_repo, self.python_executable, self.hf_model]

    @property
    def python_executable(self) -> PythonExecutable:
        if not self._python_executable:
            self._python_executable = PythonExecutable(
                GitRepo(url=self.genai_perf_repo.url, commit=self.genai_perf_repo.commit),
            )
        return self._python_executable

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        output_path = tr.output_path
        csv_files = list(output_path.rglob(CSV_FILES_PATTERN))
        json_files = list(output_path.rglob(JSON_FILES_PATTERN))
        logging.debug(f"Found CSV files: {csv_files}, JSON files: {json_files}")
        has_results = len(csv_files) > 0 and len(json_files) > 0
        if not has_results:
            return JobStatusResult(False, "No result files found in the output directory.")
        return JobStatusResult(True)
