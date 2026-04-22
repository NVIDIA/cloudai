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
from cloudai.workloads.vllm import (
    VllmArgs,
    VllmBenchCmdArgs,
    VllmCmdArgs,
    VllmSlurmCommandGenStrategy,
    VllmTestDefinition,
)
from cloudai.workloads.vllm.vllm import VLLM_BENCH_JSON_FILE, VLLM_BENCH_LOG_FILE, VLLM_SERVE_LOG_FILE


@pytest.fixture
def vllm() -> VllmTestDefinition:
    return VllmTestDefinition(
        name="vllm_test",
        description="vLLM benchmark test",
        test_template_name="Vllm",
        cmd_args=VllmCmdArgs(docker_image_url="nvcr.io/nvidia/vllm:latest", model="Qwen/Qwen3-0.6B", port=8000),
        extra_env_vars={"CUDA_VISIBLE_DEVICES": "0"},
    )


@pytest.fixture
def vllm_tr(vllm: VllmTestDefinition, tmp_path: Path) -> TestRun:
    return TestRun(test=vllm, num_nodes=1, nodes=[], output_path=tmp_path, name="vllm-job")


@pytest.fixture
def vllm_cmd_gen_strategy(vllm_tr: TestRun, slurm_system: SlurmSystem) -> VllmSlurmCommandGenStrategy:
    return VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)


@pytest.fixture
def vllm_disagg_tr(vllm: VllmTestDefinition, tmp_path: Path) -> TestRun:
    """TestRun for disaggregated mode with 4 GPUs."""
    vllm.extra_env_vars = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    vllm.cmd_args.prefill = VllmArgs()
    return TestRun(test=vllm, num_nodes=1, nodes=[], output_path=tmp_path, name="vllm-disagg-job")


@pytest.fixture
def vllm_disagg_2node_tr(vllm: VllmTestDefinition, tmp_path: Path) -> TestRun:
    """TestRun for disaggregated mode with 2 nodes and a shared local GPU view."""
    vllm.extra_env_vars = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    vllm.cmd_args.prefill = VllmArgs()
    return TestRun(test=vllm, num_nodes=2, nodes=[], output_path=tmp_path, name="vllm-disagg-2node-job")


def test_container_mounts(vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy) -> None:
    assert vllm_cmd_gen_strategy._container_mounts() == [
        f"{vllm_cmd_gen_strategy.system.hf_home_path.absolute()}:/root/.cache/huggingface"
    ]


def test_sweep_detection(vllm: VllmTestDefinition) -> None:
    assert vllm.is_dse_job is False
    vllm.cmd_args.decode.gpu_ids = ["1"]
    assert vllm.is_dse_job is True


class TestGpuDetection:
    """Tests for GPU detection logic."""

    def test_prefill_nodes_set(self, vllm_tr: TestRun, slurm_system: SlurmSystem) -> None:
        slurm_system.gpus_per_node = 4
        vllm_tr.test.extra_env_vars = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
        vllm_tr.test.cmd_args.prefill = VllmArgs(gpu_ids="0,3")
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)
        assert strategy.prefill_gpu_ids == [0, 3]

    def test_decode_nodes_set(self, vllm_tr: TestRun, slurm_system: SlurmSystem) -> None:
        slurm_system.gpus_per_node = 4
        vllm_tr.test.extra_env_vars = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
        vllm_tr.test.cmd_args.decode.gpu_ids = "1,2"
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)
        assert strategy.decode_gpu_ids == [1, 2]

    def test_multinode_disagg_uses_shared_gpu_ids_per_role(
        self, vllm_disagg_2node_tr: TestRun, slurm_system: SlurmSystem
    ) -> None:
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_disagg_2node_tr)
        assert strategy.prefill_gpu_ids == [0, 1, 2, 3]
        assert strategy.decode_gpu_ids == [0, 1, 2, 3]


class TestVllmServeCommand:
    @pytest.mark.parametrize(
        "decode_nthreads,prefill_nthreads",
        [
            (None, None),
            (4, 2),
            (None, 2),
            (4, None),
        ],
    )
    def test_nixl_threads(
        self,
        decode_nthreads: int | None,
        prefill_nthreads: int | None,
        vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy,
    ) -> None:
        tdef = cast(VllmTestDefinition, vllm_cmd_gen_strategy.test_run.test)
        tdef.cmd_args.prefill = VllmArgs(nixl_threads=prefill_nthreads)
        tdef.cmd_args.decode.nixl_threads = decode_nthreads

        commands = vllm_cmd_gen_strategy.get_serve_commands()

        assert len(commands) == 2

        prefill_cmd = " ".join(commands[0])
        assert "--nixl-threads" not in prefill_cmd
        if prefill_nthreads is not None:
            assert "kv_connector_extra_config" in prefill_cmd
            assert f'"num_threads":{prefill_nthreads}' in prefill_cmd
        else:
            assert all(arg not in prefill_cmd for arg in ["num_threads", "kv_connector_extra_config"])

        decode_cmd = " ".join(commands[1])
        assert "--nixl-threads" not in decode_cmd
        if decode_nthreads is not None:
            assert "kv_connector_extra_config" in decode_cmd
            assert f'"num_threads":{decode_nthreads}' in decode_cmd
        else:
            assert all(arg not in decode_cmd for arg in ["num_threads", "kv_connector_extra_config"])


class TestVllmBenchCommand:
    def test_get_vllm_bench_command(self, vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy) -> None:
        tdef = cast(VllmTestDefinition, vllm_cmd_gen_strategy.test_run.test)
        cmd_args = tdef.cmd_args
        bench_args = tdef.bench_cmd_args

        command = vllm_cmd_gen_strategy.get_bench_command()

        expected = [
            "vllm",
            "bench",
            "serve",
            f"--model {cmd_args.model}",
            f"--base-url http://${{NODE}}:{cmd_args.port}",
            f"--random-input-len {bench_args.random_input_len}",
            f"--random-output-len {bench_args.random_output_len}",
            f"--max-concurrency {bench_args.max_concurrency}",
            f"--num-prompts {bench_args.num_prompts}",
            f"--result-dir {vllm_cmd_gen_strategy.test_run.output_path.absolute()}",
            f"--result-filename {VLLM_BENCH_JSON_FILE}",
            "--save-result",
        ]
        assert command == expected

    def test_get_vllm_bench_command_with_extra_args(
        self, vllm: VllmTestDefinition, vllm_tr: TestRun, slurm_system: SlurmSystem
    ) -> None:
        vllm.bench_cmd_args = VllmBenchCmdArgs.model_validate({"extra1": 1, "extra-2": 2, "extra_3": 3})
        vllm_tr.test = vllm
        vllm_cmd_gen_strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)

        cmd = vllm_cmd_gen_strategy.get_bench_command()

        assert "--extra1 1" in cmd
        assert "--extra-2 2" in cmd
        assert "--extra-3 3" in cmd


class TestVllmAggregatedMode:
    """Tests for vLLM non-disaggregated mode with 1 GPU."""

    def test_get_vllm_serve_commands_single_gpu(self, vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy) -> None:
        cmd_args = vllm_cmd_gen_strategy.test_run.test.cmd_args

        commands = vllm_cmd_gen_strategy.get_serve_commands()

        assert len(commands) == 1
        assert commands[0] == ["vllm", "serve", cmd_args.model, "--host", cmd_args.host, "--port", str(cmd_args.port)]

    def test_get_vllm_serve_commands_convert_boolean_flags(
        self, vllm: VllmTestDefinition, vllm_tr: TestRun, slurm_system: SlurmSystem
    ) -> None:
        vllm.cmd_args.decode = VllmArgs.model_validate({"enable_expert_parallel": True})
        vllm_tr.test = vllm
        vllm_cmd_gen_strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)

        commands = vllm_cmd_gen_strategy.get_serve_commands()

        assert commands[0] == [
            "vllm",
            "serve",
            vllm.cmd_args.model,
            "--host",
            vllm.cmd_args.host,
            "--enable-expert-parallel",
            "--port",
            str(vllm.cmd_args.port),
        ]

    def test_generate_wait_for_health_function(self, vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy) -> None:
        cmd_args = vllm_cmd_gen_strategy.test_run.test.cmd_args

        func = vllm_cmd_gen_strategy.generate_wait_for_health_function()

        expected = f"""\
wait_for_health() {{
    local endpoint="$1"
    local timeout={cmd_args.serve_wait_seconds}
    local interval=5
    local end_time=$(($(date +%s) + timeout))

    while [ "$(date +%s)" -lt "$end_time" ]; do
        if curl -sf "$endpoint" > /dev/null 2>&1; then
            echo "Health check passed: $endpoint"
            return 0
        fi
        sleep "$interval"
    done

    echo "Timeout waiting for: $endpoint"
    return 1
}}"""

        assert func == expected

    def test_gen_srun_command_full_flow(self, vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy) -> None:
        tdef = vllm_cmd_gen_strategy.test_run.test
        cmd_args = tdef.cmd_args
        output_path = vllm_cmd_gen_strategy.test_run.output_path.absolute()
        srun_prefix = " ".join(vllm_cmd_gen_strategy.gen_srun_prefix())
        serve_cmd = " ".join(vllm_cmd_gen_strategy.get_serve_commands()[0])
        bench_cmd = " ".join(vllm_cmd_gen_strategy.get_bench_command())
        health_func = vllm_cmd_gen_strategy.generate_wait_for_health_function()

        srun_command = vllm_cmd_gen_strategy._gen_srun_command()

        expected = f"""\
cleanup() {{
    echo "Cleaning up PIDs: SERVE_PID=$SERVE_PID"
    [ -n "$SERVE_PID" ] && kill -9 $SERVE_PID 2>/dev/null
}}
trap cleanup EXIT

{health_func}

echo "Starting vLLM instances..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/{VLLM_SERVE_LOG_FILE} \\
    {serve_cmd} &
SERVE_PID=$!

NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
echo "Waiting for vLLM on $NODE to be ready..."
wait_for_health "http://${{NODE}}:{cmd_args.port}/healthcheck" || exit 1

echo "Running benchmark..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/{VLLM_BENCH_LOG_FILE} \\
    {bench_cmd}"""

        assert srun_command == expected


class TestVllmDisaggregatedMode:
    """Tests for vLLM disaggregated mode with multiple GPUs."""

    def test_prefill_gpu_ids(self, vllm_disagg_tr: TestRun, slurm_system: SlurmSystem) -> None:
        """Prefill gets first half of GPUs."""
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_disagg_tr)
        assert strategy.prefill_gpu_ids == [0, 1]

    def test_decode_gpu_ids(self, vllm_disagg_tr: TestRun, slurm_system: SlurmSystem) -> None:
        """Decode gets second half of GPUs."""
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_disagg_tr)
        assert strategy.decode_gpu_ids == [2, 3]

    def test_get_vllm_serve_commands_returns_two(self, vllm_disagg_tr: TestRun, slurm_system: SlurmSystem) -> None:
        """Disagg mode returns prefill and decode commands."""
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_disagg_tr)
        cmd_args = vllm_disagg_tr.test.cmd_args

        commands = strategy.get_serve_commands()

        assert len(commands) == 2
        prefill_cmd, decode_cmd = commands

        assert prefill_cmd == [
            "vllm",
            "serve",
            cmd_args.model,
            "--host",
            cmd_args.host,
            "--port",
            str(cmd_args.port + 100),
            "--kv-transfer-config",
            '\'{"kv_connector":"NixlConnector","kv_role":"kv_producer"}\'',
        ]
        assert decode_cmd == [
            "vllm",
            "serve",
            cmd_args.model,
            "--host",
            cmd_args.host,
            "--port",
            str(cmd_args.port + 200),
            "--kv-transfer-config",
            '\'{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}\'',
        ]

    def test_get_helper_command(self, vllm_disagg_tr: TestRun, slurm_system: SlurmSystem) -> None:
        """Helper command routes to prefill and decode ports."""
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_disagg_tr)
        cmd_args = vllm_disagg_tr.test.cmd_args

        command = strategy.get_helper_command()

        assert command == [
            "python3",
            cmd_args.proxy_script,
            "--host",
            cmd_args.host,
            "--port",
            str(cmd_args.port),
            "--prefiller-hosts",
            "${PREFILL_NODE}",
            "--prefiller-ports",
            str(cmd_args.port + 100),
            "--decoder-hosts",
            "${DECODE_NODE}",
            "--decoder-ports",
            str(cmd_args.port + 200),
        ]

    def test_gen_srun_command_disagg_flow(self, vllm_disagg_tr: TestRun, slurm_system: SlurmSystem) -> None:
        """Disagg mode starts prefill, decode, and helper, waits for health checks."""
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_disagg_tr)
        cmd_args = vllm_disagg_tr.test.cmd_args
        output_path = vllm_disagg_tr.output_path.absolute()
        srun_prefix = " ".join(strategy.gen_srun_prefix())
        prefill_cmd, decode_cmd = strategy.get_serve_commands()
        helper_cmd = strategy.get_helper_command()
        bench_cmd = " ".join(strategy.get_bench_command())
        health_func = strategy.generate_wait_for_health_function()
        prefill_gpus = ",".join(str(g) for g in strategy.prefill_gpu_ids)
        decode_gpus = ",".join(str(g) for g in strategy.decode_gpu_ids)
        prefill_env = (
            f'env CUDA_VISIBLE_DEVICES="{prefill_gpus}" '
            'VLLM_NIXL_SIDE_CHANNEL_HOST="${PREFILL_NODE}" '
            'VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_NIXL_PORT"'
        )
        decode_env = (
            f'env CUDA_VISIBLE_DEVICES="{decode_gpus}" '
            'VLLM_NIXL_SIDE_CHANNEL_HOST="${DECODE_NODE}" '
            'VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_NIXL_PORT"'
        )

        srun_command = strategy._gen_srun_command()

        expected = f"""\
cleanup() {{
    echo "Cleaning up PIDs: PREFILL_PID=$PREFILL_PID DECODE_PID=$DECODE_PID HELPER_PID=$HELPER_PID"
    [ -n "$PREFILL_PID" ] && kill -9 $PREFILL_PID 2>/dev/null
    [ -n "$DECODE_PID" ] && kill -9 $DECODE_PID 2>/dev/null
    [ -n "$HELPER_PID" ] && kill -9 $HELPER_PID 2>/dev/null
}}
trap cleanup EXIT

{health_func}

PORT_OFFSET=$((SLURM_JOB_ID % 1000))
PREFILL_NIXL_PORT=$((5557 + PORT_OFFSET))
DECODE_NIXL_PORT=$((5557 + PORT_OFFSET + {len(strategy.gpu_ids)}))

NODES=( $(scontrol show hostname $SLURM_JOB_NODELIST) )
PREFILL_NODE=${{NODES[0]}}
DECODE_NODE=${{NODES[1]:-${{PREFILL_NODE}}}}
if [ -z "$PREFILL_NODE" ]; then
    echo "Failed to resolve allocated nodes for disaggregated vLLM"
    exit 1
fi
echo "Node roles: prefill=$PREFILL_NODE decode=$DECODE_NODE"

echo "Starting vLLM instances..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/vllm-prefill.log \\
    {prefill_env} {" ".join(prefill_cmd)} &
PREFILL_PID=$!

{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/vllm-decode.log \\
    {decode_env} {" ".join(decode_cmd)} &
DECODE_PID=$!

echo "Waiting for vLLM on $PREFILL_NODE and $DECODE_NODE to be ready..."
wait_for_health "http://${{PREFILL_NODE}}:{cmd_args.port + 100}/health" || exit 1
wait_for_health "http://${{DECODE_NODE}}:{cmd_args.port + 200}/health" || exit 1

echo "Starting router..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/vllm-router.log \\
    {" ".join(helper_cmd)} &
HELPER_PID=$!

echo "Waiting for vLLM on $PREFILL_NODE server to be ready..."
wait_for_health "http://${{PREFILL_NODE}}:{cmd_args.port}/healthcheck" || exit 1

echo "Running benchmark..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/{VLLM_BENCH_LOG_FILE} \\
    {bench_cmd}"""

        assert srun_command == expected

    def test_gen_srun_command_disagg_two_nodes_flow(
        self, vllm_disagg_2node_tr: TestRun, slurm_system: SlurmSystem
    ) -> None:
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_disagg_2node_tr)

        srun_command = strategy._gen_srun_command()

        assert "PREFILL_NODE=${NODES[0]}" in srun_command
        assert "DECODE_NODE=${NODES[1]:-${PREFILL_NODE}}" in srun_command
        assert srun_command.count("--relative=0 -N1") == 3
        assert srun_command.count("--relative=1 -N1") == 1
        assert (
            'env CUDA_VISIBLE_DEVICES="0,1,2,3" VLLM_NIXL_SIDE_CHANNEL_HOST="${PREFILL_NODE}" '
            'VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_NIXL_PORT"'
        ) in srun_command
        assert (
            'env CUDA_VISIBLE_DEVICES="0,1,2,3" VLLM_NIXL_SIDE_CHANNEL_HOST="${DECODE_NODE}" '
            'VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_NIXL_PORT"'
        ) in srun_command
        assert 'wait_for_health "http://${PREFILL_NODE}:8100/health"' in srun_command
        assert 'wait_for_health "http://${DECODE_NODE}:8200/health"' in srun_command
        assert "--prefiller-hosts ${PREFILL_NODE}" in srun_command
        assert "--decoder-hosts ${DECODE_NODE}" in srun_command
        assert "--base-url http://${PREFILL_NODE}:8000" in srun_command

    def test_disagg_more_than_two_nodes_is_rejected(self, vllm_disagg_tr: TestRun, slurm_system: SlurmSystem) -> None:
        vllm_disagg_tr.num_nodes = 3
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_disagg_tr)

        with pytest.raises(ValueError, match="supports only 1 or 2 nodes"):
            _ = strategy._gen_srun_command()
