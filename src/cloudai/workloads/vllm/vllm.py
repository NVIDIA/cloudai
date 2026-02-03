# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from cloudai.core import DockerImage, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition

VLLM_SERVE_LOG_FILE = "vllm-serve.log"
VLLM_BENCH_LOG_FILE = "vllm-bench.log"


class VllmCmdArgs(CmdArgs):
    """vLLM serve command arguments."""

    docker_image_url: str
    port: int = 8000
    vllm_serve_wait_seconds: int = 300
    model: str = "Qwen/Qwen3-0.6B"


class VllmBenchCmdArgs(CmdArgs):
    """vLLM bench serve command arguments."""

    random_input_len: int = 16
    random_output_len: int = 128
    max_concurrency: int = 16
    num_prompts: int = 30


class VllmTestDefinition(TestDefinition):
    """Test object for vLLM."""

    cmd_args: VllmCmdArgs
    bench_cmd_args: VllmBenchCmdArgs = VllmBenchCmdArgs()

    _docker_image: DockerImage | None = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [*self.git_repos, self.docker_image]

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        log_path = tr.output_path / VLLM_BENCH_LOG_FILE
        if not log_path.is_file():
            return JobStatusResult(is_successful=False, error_message=f"vLLM bench log not found in {tr.output_path}.")

        with log_path.open("r") as f:
            for line in f:
                if "============ Serving Benchmark Result ============" in line:
                    return JobStatusResult(is_successful=True)

        return JobStatusResult(
            is_successful=False, error_message=f"vLLM bench log does not contain benchmark result in {tr.output_path}."
        )
