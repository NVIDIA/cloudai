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

from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import Field
from rich.table import Table

from cloudai.core import GitRepo, TestRun
from cloudai.workloads.common.llm_serving import (
    LLMServingArgs,
    LLMServingBenchReport,
    LLMServingCmdArgs,
    LLMServingReportGenerationStrategy,
    LLMServingTestDefinition,
    all_gpu_ids,
)


class FakeLLMArgs(LLMServingArgs):
    gpu_ids: str | None = None
    internal_only: str | None = None

    @property
    def serve_args_exclude(self) -> set[str]:
        return {"gpu_ids", "internal_only"}


class FakeLLMCmdArgs(LLMServingCmdArgs):
    docker_image_url: str = "image:latest"
    model: str = "test/model"
    prefill: FakeLLMArgs | None = None
    decode: FakeLLMArgs = Field(default_factory=FakeLLMArgs)


class FakeLLMTestDefinition(LLMServingTestDefinition[FakeLLMCmdArgs]):
    cmd_args: FakeLLMCmdArgs


class PlainLLMArgs(LLMServingArgs):
    pass


class FakeBenchReport(LLMServingBenchReport):
    request_throughput: float

    @property
    def throughput(self) -> float:
        return self.request_throughput


class FakeReportStrategy(LLMServingReportGenerationStrategy[FakeLLMTestDefinition, FakeBenchReport]):
    def __init__(self, system: Any, tr: TestRun, result: FakeBenchReport | None) -> None:
        super().__init__(system, tr)
        self._result = result

    @property
    def result_file_name(self) -> str:
        return "fake-results.json"

    @property
    def report_title(self) -> str:
        return "Fake LLM Results"

    def parse_output(self, path):  # type: ignore[override]
        return self._result

    def all_gpu_ids(self, tdef, gpus_per_node: int | None) -> list[int]:  # type: ignore[override]
        return [0]


@pytest.fixture
def llm_tdef() -> FakeLLMTestDefinition:
    return FakeLLMTestDefinition(
        name="llm_test",
        description="LLM benchmark test",
        test_template_name="llm",
        cmd_args=FakeLLMCmdArgs(),
    )


def make_tdef(
    prefill_gpu_ids: str | None = None,
    decode_gpu_ids: str | None = None,
    create_prefill: bool = False,
    git_repos: list[GitRepo] | None = None,
) -> FakeLLMTestDefinition:
    prefill: FakeLLMArgs | None = None
    if create_prefill or prefill_gpu_ids is not None:
        prefill = FakeLLMArgs(gpu_ids=prefill_gpu_ids)

    return FakeLLMTestDefinition(
        name="llm_test",
        description="LLM benchmark test",
        test_template_name="llm",
        git_repos=git_repos or [],
        cmd_args=FakeLLMCmdArgs(prefill=prefill, decode=FakeLLMArgs(gpu_ids=decode_gpu_ids)),
    )


class TestAllGpuIds:
    @pytest.mark.parametrize("cuda_visible_devices", ["0", "0,1,2,3", "0,1,2,3,4,5,6,7"])
    def test_from_cuda_visible_devices(self, llm_tdef: FakeLLMTestDefinition, cuda_visible_devices: str) -> None:
        llm_tdef.extra_env_vars = {"CUDA_VISIBLE_DEVICES": cuda_visible_devices}

        assert all_gpu_ids(cast(Any, llm_tdef), 8) == [int(gpu_id) for gpu_id in cuda_visible_devices.split(",")]

    @pytest.mark.parametrize("gpus_per_node", [None, 1, 8])
    def test_fallback_to_system_gpu_count(self, llm_tdef: FakeLLMTestDefinition, gpus_per_node: int | None) -> None:
        llm_tdef.extra_env_vars = {}

        assert all_gpu_ids(cast(Any, llm_tdef), gpus_per_node) == list(range(gpus_per_node or 1))

    def test_prefill_and_decode_gpu_ids_override_cuda_visible_devices(self, llm_tdef: FakeLLMTestDefinition) -> None:
        llm_tdef.extra_env_vars = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
        llm_tdef.cmd_args.prefill = FakeLLMArgs(gpu_ids="4")
        llm_tdef.cmd_args.decode.gpu_ids = "5"

        assert all_gpu_ids(cast(Any, llm_tdef), 4) == [4, 5]


class TestLLMServingArgsBehavior:
    def test_default_serve_args_exclude_is_empty(self) -> None:
        assert PlainLLMArgs.model_validate({"some_flag": "", "gpu_ids": "0"}).serve_args == [
            "--some-flag",
            "--gpu-ids",
            "0",
        ]

    def test_serve_args_empty_excludes_internal_fields(self) -> None:
        assert FakeLLMArgs().serve_args == []
        assert FakeLLMArgs(gpu_ids="0", internal_only="skip").serve_args == []

    def test_serve_args_serializes_flags_and_values(self) -> None:
        assert FakeLLMArgs.model_validate(
            {"some_flag": "", "some_arg": "value", "zero_value": 0, "none_value": None}
        ).serve_args == [
            "--some-flag",
            "--some-arg",
            "value",
            "--zero-value",
            "0",
        ]


class TestLLMServingTestDefinitionBehavior:
    def test_shared_installables_include_git_repos_docker_image_and_hf_model(
        self, llm_tdef: FakeLLMTestDefinition
    ) -> None:
        repo = GitRepo(url="./repo", commit="commit")
        llm_tdef.git_repos = [repo]

        assert llm_tdef.installables == [repo, llm_tdef.docker_image, llm_tdef.hf_model]

    def test_docker_image_and_hf_model_are_cached(self, llm_tdef: FakeLLMTestDefinition) -> None:
        assert llm_tdef.docker_image is llm_tdef.docker_image
        assert llm_tdef.hf_model is llm_tdef.hf_model

    @pytest.mark.parametrize(
        ("prefill_gpu_ids", "decode_gpu_ids"),
        [("0,1", "0,1"), (None, None), (None, "11,42")],
    )
    def test_valid_gpu_ids_configuration(
        self,
        prefill_gpu_ids: str | None,
        decode_gpu_ids: str | None,
    ) -> None:
        tdef = make_tdef(prefill_gpu_ids, decode_gpu_ids)

        if prefill_gpu_ids is not None:
            assert tdef.cmd_args.prefill
            assert tdef.cmd_args.prefill.gpu_ids == prefill_gpu_ids

        assert tdef.cmd_args.decode.gpu_ids == decode_gpu_ids

    @pytest.mark.parametrize(("prefill_gpu_ids", "decode_gpu_ids"), [("0,1", None), (None, "0,1")])
    def test_invalid_gpu_ids_configuration(
        self,
        prefill_gpu_ids: str | None,
        decode_gpu_ids: str | None,
    ) -> None:
        with pytest.raises(
            ValueError,
            match=r"Both prefill and decode gpu_ids must be set or both must be None\.",
        ):
            make_tdef(prefill_gpu_ids, decode_gpu_ids, True)


def test_generate_report_uses_shared_table_builder(
    llm_tdef: FakeLLMTestDefinition, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tr = TestRun(name="llm", test=llm_tdef, num_nodes=1, nodes=[], output_path=tmp_path)
    strategy = FakeReportStrategy(
        SimpleNamespace(gpus_per_node=1),
        tr,
        FakeBenchReport(
            num_prompts=10,
            completed=9,
            mean_ttft_ms=1.0,
            median_ttft_ms=2.0,
            p99_ttft_ms=3.0,
            mean_tpot_ms=4.0,
            median_tpot_ms=5.0,
            p99_tpot_ms=6.0,
            max_concurrency=2,
            request_throughput=8.0,
        ),
    )
    printed: list[Table] = []

    def capture_print(_console_self: object, table: Table) -> None:
        printed.append(table)

    monkeypatch.setattr("cloudai.workloads.common.llm_serving.Console.print", capture_print)

    strategy.generate_report()

    assert len(printed) == 1
    assert printed[0].title == f"Fake LLM Results ({tmp_path})"
