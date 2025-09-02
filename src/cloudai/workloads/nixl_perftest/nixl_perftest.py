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

from typing import Literal, Optional

from pydantic import Field, model_validator

from cloudai.core import CmdArgs, DockerImage, Installable, TestDefinition, TestRun


class MatgenCmdArgs(CmdArgs):
    """Command args for matgen script."""

    ppn: int | None = None


class NixlPerftestCmdArgs(CmdArgs):
    """CmdArgs for NixlPerftestTestDefinition."""

    docker_image_url: str

    subtest: Literal["sequential-ct-perftest"]
    perftest_script: str = "/workspace/nixl/benchmark/kvbench/main.py"
    matgen_script: str = "/workspace/nixl/benchmark/kvbench/test/inference_workload_matgen.py"
    python_executable: str = "/workspace/nixl/.venv/bin/python"
    etcd_path: str = "etcd"
    wait_etcd_for: int = 60

    num_user_requests: int | list[int]
    batch_size: int | list[int]
    num_prefill_nodes: int | list[int]
    num_decode_nodes: int | list[int]
    isl_mean: int | list[int] | None = None
    isl_scale: int | list[int] | None = None
    prefill_tp: int | list[int] = 1
    prefill_pp: int | list[int] = 1
    prefill_cp: int | list[int] = 1
    decode_tp: int | list[int] = 1
    decode_pp: int | list[int] = 1
    decode_cp: int | list[int] = 1

    # model or model configuration
    model: str | list[str] | None = None
    hidden_size: int | None = None
    num_layers: int | None = None
    num_heads: int | None = None
    num_kv_heads: int | None = None
    dtype_size: int | None = None

    matgen_args: MatgenCmdArgs = Field(default_factory=MatgenCmdArgs)

    @model_validator(mode="after")
    def model_vs_custom(self):
        if self.model is None and (
            self.hidden_size is None
            or self.num_layers is None
            or self.num_heads is None
            or self.num_kv_heads is None
            or self.dtype_size is None
        ):
            raise ValueError(
                "If 'model' is None, 'hidden_size', 'num_layers', 'num_heads', 'num_kv_heads', and 'dtype_size' "
                "must be specified."
            )

        if self.model is not None and (
            self.hidden_size is not None
            or self.num_layers is not None
            or self.num_heads is not None
            or self.num_kv_heads is not None
            or self.dtype_size is not None
        ):
            raise ValueError(
                "If 'model' is specified, 'hidden_size', 'num_layers', 'num_heads', 'num_kv_heads', and 'dtype_size' "
                "must be None."
            )

        return self


class NixlPerftestTestDefinition(TestDefinition):
    """TestDefinition for NixlPerftest."""

    _docker_image: Optional[DockerImage] = None
    cmd_args: NixlPerftestCmdArgs

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [*self.git_repos, self.docker_image]

    def constraint_check(self, tr: TestRun) -> bool:
        decode_tp = int(tr.test.test_definition.cmd_args.decode_tp)
        decode_nodes = int(tr.test.test_definition.cmd_args.num_decode_nodes)
        prefill_tp = int(tr.test.test_definition.cmd_args.prefill_tp)
        prefill_nodes = int(tr.test.test_definition.cmd_args.num_prefill_nodes)

        return decode_tp / decode_nodes == prefill_tp / prefill_nodes
