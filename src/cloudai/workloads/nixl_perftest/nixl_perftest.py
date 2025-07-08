# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES All rights reserved.

from typing import Literal, Optional

from pydantic import model_validator

from cloudai.core import CmdArgs, DockerImage, Installable, PythonExecutable, TestDefinition


class NixlPerftestCmdArgs(CmdArgs):
    """CmdArgs for NixlPerftestTestDefinition."""

    docker_image_url: str

    subtest: Literal["sequential-ct-perftest"]
    perftest_script: str
    matgen_script: str
    python_executable: str

    num_user_requests: int
    batch_size: int
    num_prefill_nodes: int
    num_decode_nodes: int
    isl_mean: float | None = None
    isl_scale: float | None = None
    prefill_tp: int | None = None
    prefill_pp: int | None = None
    prefill_cp: int | None = None
    decode_tp: int | None = None
    decode_pp: int | None = None
    decode_cp: int | None = None
    model: str | None = None
    hidden_size: int | None = None
    num_layers: int | None = None
    num_heads: int | None = None
    num_kv_heads: int | None = None
    dtype_size: int | None = None

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
