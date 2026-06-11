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
from typing import cast

import pytest

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.sglang import (
    SglangArgs,
    SglangCmdArgs,
    SglangSemanticEvalCmdArgs,
    SglangSlurmCommandGenStrategy,
    SglangTestDefinition,
)


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


@pytest.fixture
def sglang_disagg_2node_tr(sglang: SglangTestDefinition, tmp_path: Path) -> TestRun:
    sglang.extra_env_vars = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    sglang.cmd_args.prefill = SglangArgs()
    return TestRun(test=sglang, num_nodes=2, nodes=[], output_path=tmp_path, name="sglang-disagg-2node-job")


def test_container_mounts(sglang_cmd_gen_strategy: SglangSlurmCommandGenStrategy) -> None:
    assert sglang_cmd_gen_strategy._container_mounts() == [
        f"{sglang_cmd_gen_strategy.system.hf_home_path.absolute()}:/root/.cache/huggingface"
    ]


class TestGpuDetection:
    def test_aggregated_gpu_ids_from_decode_config(self, sglang_tr: TestRun, slurm_system: SlurmSystem) -> None:
        tdef = cast(SglangTestDefinition, sglang_tr.test)
        tdef.extra_env_vars = {}
        tdef.cmd_args.decode.gpu_ids = "0,1,2,3"
        strategy = SglangSlurmCommandGenStrategy(slurm_system, sglang_tr)

        assert strategy.gpu_ids == [0, 1, 2, 3]
        assert 'env CUDA_VISIBLE_DEVICES="0,1,2,3"' in strategy._gen_srun_command()

    def test_multinode_disagg_uses_shared_gpu_ids_per_role(
        self, sglang_disagg_2node_tr: TestRun, slurm_system: SlurmSystem
    ) -> None:
        strategy = SglangSlurmCommandGenStrategy(slurm_system, sglang_disagg_2node_tr)
        assert strategy.prefill_gpu_ids == [0, 1, 2, 3]
        assert strategy.decode_gpu_ids == [0, 1, 2, 3]


def test_get_sglang_semantic_eval_command_defaults(sglang_cmd_gen_strategy: SglangSlurmCommandGenStrategy):
    sglang_test = cast(SglangTestDefinition, sglang_cmd_gen_strategy.test_run.test)
    sglang_test.semantic_eval_cmd_args = SglangSemanticEvalCmdArgs()

    command = sglang_cmd_gen_strategy.get_semantic_eval_command()

    assert command == [
        "python3 -m sglang.test.run_eval",
        "--host ${NODE} --port 8000 --eval-name gsm8k --num-examples 200 --num-threads 128 --model Qwen/Qwen3-8B",
    ]


def test_get_sglang_semantic_eval_command_supports_custom_entrypoint_and_cli(
    sglang_cmd_gen_strategy: SglangSlurmCommandGenStrategy,
):
    sglang_test = cast(SglangTestDefinition, sglang_cmd_gen_strategy.test_run.test)
    sglang_test.semantic_eval_cmd_args = SglangSemanticEvalCmdArgs(
        entrypoint="python3 /custom/semantic_eval.py",
        cli="--num-questions 200 --data-path {result_dir}/gsm8k.jsonl --seen {url}",
    )

    command = sglang_cmd_gen_strategy.get_semantic_eval_command()

    assert command is not None
    assert command[0] == "python3 /custom/semantic_eval.py"
    assert command[-1] == (
        f"--num-questions 200 --data-path {sglang_cmd_gen_strategy.test_run.output_path.absolute()}/gsm8k.jsonl "
        "--seen ${NODE}:8000"
    )


def test_gen_srun_command_contains_sglang_semantic_eval(sglang_cmd_gen_strategy: SglangSlurmCommandGenStrategy):
    sglang_test = cast(SglangTestDefinition, sglang_cmd_gen_strategy.test_run.test)
    sglang_test.semantic_eval_cmd_args = SglangSemanticEvalCmdArgs()

    srun_command = sglang_cmd_gen_strategy._gen_srun_command()

    assert "Running benchmark..." in srun_command
    assert "Running semantic validation..." in srun_command
    assert (
        "--output=" + str((sglang_cmd_gen_strategy.test_run.output_path / "sglang-semantic-eval.log").absolute())
        in srun_command
    )
    assert "python3 -m sglang.test.run_eval --host ${NODE} --port 8000" in srun_command


def test_gen_srun_command_contains_sglang_semantic_eval_in_disagg(
    sglang_disagg_tr: TestRun, slurm_system: SlurmSystem
) -> None:
    sglang_test = cast(SglangTestDefinition, sglang_disagg_tr.test)
    sglang_test.semantic_eval_cmd_args = SglangSemanticEvalCmdArgs()
    strategy = SglangSlurmCommandGenStrategy(slurm_system, sglang_disagg_tr)

    srun_command = strategy._gen_srun_command()

    assert "Running semantic validation..." in srun_command
    assert "python3 -m sglang.test.run_eval --host ${PREFILL_NODE} --port 8000" in srun_command


def test_disagg_more_than_two_nodes_is_rejected(sglang_disagg_tr: TestRun, slurm_system: SlurmSystem) -> None:
    sglang_disagg_tr.num_nodes = 3
    strategy = SglangSlurmCommandGenStrategy(slurm_system, sglang_disagg_tr)

    with pytest.raises(ValueError, match=r"requires both prefill\.num_nodes and decode\.num_nodes"):
        _ = strategy._gen_srun_command()


def test_gen_srun_command_disagg_four_nodes_uses_separate_sglang_distributed_launches(
    sglang_disagg_tr: TestRun, slurm_system: SlurmSystem
) -> None:
    tdef = cast(SglangTestDefinition, sglang_disagg_tr.test)
    assert tdef.cmd_args.prefill is not None
    tdef.cmd_args.prefill.num_nodes = 2
    tdef.cmd_args.decode.num_nodes = 2
    sglang_disagg_tr.num_nodes = 4
    strategy = SglangSlurmCommandGenStrategy(slurm_system, sglang_disagg_tr)

    srun_command = strategy._gen_srun_command()

    assert 'PREFILL_NODES=( "${NODES[@]:0:2}" )' in srun_command
    assert 'DECODE_NODES=( "${NODES[@]:2:2}" )' in srun_command
    assert "export PREFILL_DIST_INIT_PORT=$((20000 + PORT_OFFSET))" in srun_command
    assert "export DECODE_DIST_INIT_PORT=$((21000 + PORT_OFFSET))" in srun_command
    assert '--nodelist="${PREFILL_NODELIST}" --nodes=2 --ntasks=2 --ntasks-per-node=1' in srun_command
    assert '--nodelist="${DECODE_NODELIST}" --nodes=2 --ntasks=2 --ntasks-per-node=1' in srun_command
    assert (
        '--dist-init-addr "${PREFILL_NODE}:${PREFILL_DIST_INIT_PORT}" --nnodes 2 --node-rank "$SLURM_PROCID"'
        in srun_command
    )
    assert (
        '--dist-init-addr "${DECODE_NODE}:${DECODE_DIST_INIT_PORT}" --nnodes 2 --node-rank "$SLURM_PROCID"'
        in srun_command
    )


def test_custom_bash_string_wraps_aggregated_serve_and_benchmark(
    sglang_cmd_gen_strategy: SglangSlurmCommandGenStrategy,
) -> None:
    tdef = cast(SglangTestDefinition, sglang_cmd_gen_strategy.test_run.test)
    tdef.custom_bash = "echo setup"

    srun_command = sglang_cmd_gen_strategy._gen_srun_command()

    assert srun_command.count("bash -c ") == 2
    assert "echo setup; exec env CUDA_VISIBLE_DEVICES" in srun_command
    assert "python3 -m sglang.launch_server" in srun_command
    assert "echo setup; exec python3 -m sglang.bench_serving" in srun_command


def test_custom_bash_regex_can_target_sglang_disaggregated_commands(
    sglang_disagg_tr: TestRun, slurm_system: SlurmSystem
) -> None:
    tdef = cast(SglangTestDefinition, sglang_disagg_tr.test)
    tdef.custom_bash = {
        "sglang.launch_server.*prefill": "echo prefill setup",
        "sglang.launch_server.*decode": "echo decode setup",
        "sglang_router.launch_router": "echo router setup",
        "sglang.bench_serving": "echo bench setup",
    }
    strategy = SglangSlurmCommandGenStrategy(slurm_system, sglang_disagg_tr)

    srun_command = strategy._gen_srun_command()

    assert srun_command.count("bash -c ") == 4
    assert "echo prefill setup; exec env" in srun_command
    assert "echo decode setup; exec env" in srun_command
    assert "echo router setup; exec python3 -m sglang_router.launch_router" in srun_command
    assert "echo bench setup; exec python3 -m sglang.bench_serving" in srun_command
