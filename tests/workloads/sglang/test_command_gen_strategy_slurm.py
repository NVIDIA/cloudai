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

from pathlib import Path

import pytest

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.sglang import SglangArgs, SglangCmdArgs, SglangSlurmCommandGenStrategy, SglangTestDefinition
from cloudai.workloads.sglang.sglang import SGLANG_BENCH_LOG_FILE


@pytest.fixture
def sglang() -> SglangTestDefinition:
    return SglangTestDefinition(
        name="sglang_test",
        description="SGLang benchmark test",
        test_template_name="sglang",
        cmd_args=SglangCmdArgs(docker_image_url="docker.io/lmsysorg/sglang:dev", model="Qwen/Qwen3-8B", port=8000),
        extra_env_vars={"CUDA_VISIBLE_DEVICES": "0"},
    )


@pytest.fixture
def sglang_tr(sglang: SglangTestDefinition, tmp_path: Path) -> TestRun:
    return TestRun(test=sglang, num_nodes=1, nodes=[], output_path=tmp_path, name="sglang-job")


@pytest.fixture
def sglang_cmd_gen_strategy(sglang_tr: TestRun, slurm_system: SlurmSystem) -> SglangSlurmCommandGenStrategy:
    return SglangSlurmCommandGenStrategy(slurm_system, sglang_tr)


@pytest.fixture
def sglang_disagg_tr(sglang: SglangTestDefinition, tmp_path: Path) -> TestRun:
    sglang.extra_env_vars = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    sglang.cmd_args.prefill = SglangArgs()
    return TestRun(test=sglang, num_nodes=1, nodes=[], output_path=tmp_path, name="sglang-disagg-job")


def test_container_mounts(sglang_cmd_gen_strategy: SglangSlurmCommandGenStrategy) -> None:
    assert sglang_cmd_gen_strategy._container_mounts() == [
        f"{sglang_cmd_gen_strategy.system.hf_home_path.absolute()}:/root/.cache/huggingface"
    ]


class TestGpuDetection:
    @pytest.mark.parametrize("cuda_visible_devices", ["0", "0,1,2,3", "0,1,2,3,4,5,6,7"])
    def test_gpu_ids_from_cuda_visible_devices(
        self, cuda_visible_devices: str, sglang_tr: TestRun, slurm_system: SlurmSystem
    ) -> None:
        sglang_tr.test.extra_env_vars = {"CUDA_VISIBLE_DEVICES": cuda_visible_devices}
        strategy = SglangSlurmCommandGenStrategy(slurm_system, sglang_tr)
        assert strategy.gpu_ids == [int(gpu_id) for gpu_id in cuda_visible_devices.split(",")]


def test_get_sglang_serve_commands_aggregated(sglang_cmd_gen_strategy: SglangSlurmCommandGenStrategy) -> None:
    cmd_args = sglang_cmd_gen_strategy.test_run.test.cmd_args
    commands = sglang_cmd_gen_strategy.get_sglang_serve_commands()

    assert len(commands) == 1
    assert commands[0] == [
        "python3",
        "-m",
        cmd_args.serve_module,
        "--model-path",
        cmd_args.model,
        "--host",
        "0.0.0.0",
        "--port",
        str(cmd_args.port),
    ]


def test_get_sglang_serve_commands_disagg(sglang_disagg_tr: TestRun, slurm_system: SlurmSystem) -> None:
    strategy = SglangSlurmCommandGenStrategy(slurm_system, sglang_disagg_tr)

    commands = strategy.get_sglang_serve_commands()

    assert len(commands) == 2
    prefill_cmd, decode_cmd = commands
    assert "--disaggregation-mode" in prefill_cmd
    assert "prefill" in prefill_cmd
    assert str(strategy.prefill_port) in prefill_cmd

    assert "--disaggregation-mode" in decode_cmd
    assert "decode" in decode_cmd
    assert str(strategy.decode_port) in decode_cmd


def test_get_sglang_bench_command_adds_pd_separated_in_disagg(
    sglang_disagg_tr: TestRun, slurm_system: SlurmSystem
) -> None:
    strategy = SglangSlurmCommandGenStrategy(slurm_system, sglang_disagg_tr)

    command = strategy.get_sglang_bench_command()

    assert "--pd-separated" in command


def test_gen_srun_command_contains_expected_flow(sglang_disagg_tr: TestRun, slurm_system: SlurmSystem) -> None:
    strategy = SglangSlurmCommandGenStrategy(slurm_system, sglang_disagg_tr)

    srun_command = strategy._gen_srun_command()

    assert "Starting SGLang instances" in srun_command
    assert "Starting router" in srun_command
    assert f"--output={strategy.test_run.output_path.absolute()}/{SGLANG_BENCH_LOG_FILE}" in srun_command
