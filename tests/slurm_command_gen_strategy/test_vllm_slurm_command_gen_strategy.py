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
from cloudai.workloads.vllm import VllmCmdArgs, VllmSlurmCommandGenStrategy, VllmTestDefinition
from cloudai.workloads.vllm.vllm import VLLM_BENCH_LOG_FILE, VLLM_SERVE_LOG_FILE


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
    return TestRun(test=vllm, num_nodes=1, nodes=[], output_path=tmp_path, name="vllm-disagg-job")


class TestGpuDetection:
    """Tests for GPU detection logic."""

    @pytest.mark.parametrize("cuda_visible_devices", ["0", "0,1,2,3", "0,1,2,3,4,5,6,7"])
    def test_gpu_ids_from_cuda_visible_devices_single(
        self, cuda_visible_devices: str, vllm_tr: TestRun, slurm_system: SlurmSystem
    ) -> None:
        vllm_tr.test.extra_env_vars = {"CUDA_VISIBLE_DEVICES": cuda_visible_devices}
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)
        assert strategy.gpu_ids == [int(gpu_id) for gpu_id in cuda_visible_devices.split(",")]

    @pytest.mark.parametrize("gpus_per_node", [None, 1, 8])
    def test_gpu_ids_fallback_to_system(
        self, gpus_per_node: int | None, vllm_tr: TestRun, slurm_system: SlurmSystem
    ) -> None:
        vllm_tr.test.extra_env_vars = {}
        slurm_system.gpus_per_node = gpus_per_node

        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)

        assert strategy.gpu_ids == list(range(gpus_per_node or 1))


class TestServeExtraArgs:
    """Tests for serve_extra_args property."""

    def test_serve_extra_args_empty_by_default(self) -> None:
        """Default cmd_args produces empty extra args (all fields excluded)."""
        tdef = VllmTestDefinition(
            name="vllm",
            description="test",
            test_template_name="Vllm",
            cmd_args=VllmCmdArgs(docker_image_url="image:latest"),
        )
        assert tdef.serve_extra_args == []

    def test_serve_extra_args_with_custom_fields(self) -> None:
        """Extra fields in cmd_args appear in serve_extra_args."""
        tdef = VllmTestDefinition(
            name="vllm",
            description="test",
            test_template_name="Vllm",
            cmd_args=VllmCmdArgs.model_validate(
                {
                    "docker_image_url": "image:latest",
                    "tensor_parallel_size": 4,
                    "max_model_len": 8192,
                }
            ),
        )
        assert tdef.serve_extra_args == [
            "--tensor-parallel-size",
            "4",
            "--max-model-len",
            "8192",
        ]

    def test_serve_extra_args_underscore_to_dash(self) -> None:
        """Underscores in field names are converted to dashes."""
        tdef = VllmTestDefinition(
            name="vllm",
            description="test",
            test_template_name="Vllm",
            cmd_args=VllmCmdArgs.model_validate(
                {
                    "docker_image_url": "image:latest",
                    "some_long_arg": "value",
                }
            ),
        )
        assert "--some-long-arg" in tdef.serve_extra_args


class TestVllmAggregatedMode:
    """Tests for vLLM non-disaggregated mode with 1 GPU."""

    def test_get_vllm_serve_commands_single_gpu(self, vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy) -> None:
        cmd_args = vllm_cmd_gen_strategy.test_run.test.cmd_args

        commands = vllm_cmd_gen_strategy.get_vllm_serve_commands()

        assert len(commands) == 1
        assert commands[0] == ["vllm", "serve", cmd_args.model, "--port", str(cmd_args.port)]

    def test_generate_wait_for_health_function(self, vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy) -> None:
        cmd_args = vllm_cmd_gen_strategy.test_run.test.cmd_args

        func = vllm_cmd_gen_strategy.generate_wait_for_health_function()

        expected = f"""\
wait_for_health() {{
    local endpoint="$1"
    local timeout={cmd_args.vllm_serve_wait_seconds}
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

    def test_get_vllm_bench_command(self, vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy) -> None:
        tdef = cast(VllmTestDefinition, vllm_cmd_gen_strategy.test_run.test)
        cmd_args = tdef.cmd_args
        bench_args = tdef.bench_cmd_args

        command = " ".join(vllm_cmd_gen_strategy.get_vllm_bench_command())

        expected = (
            f"vllm bench serve --model {cmd_args.model} "
            f"--base-url http://${{NODE}}:{cmd_args.port} "
            f"--random-input-len {bench_args.random_input_len} "
            f"--random-output-len {bench_args.random_output_len} "
            f"--max-concurrency {bench_args.max_concurrency} "
            f"--num-prompts {bench_args.num_prompts}"
        )
        assert command == expected

    def test_gen_srun_command_full_flow(self, vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy) -> None:
        tdef = vllm_cmd_gen_strategy.test_run.test
        cmd_args = tdef.cmd_args
        output_path = vllm_cmd_gen_strategy.test_run.output_path.absolute()
        srun_prefix = " ".join(vllm_cmd_gen_strategy.gen_srun_prefix())
        serve_cmd = " ".join(vllm_cmd_gen_strategy.get_vllm_serve_commands()[0])
        bench_cmd = " ".join(vllm_cmd_gen_strategy.get_vllm_bench_command())
        health_func = vllm_cmd_gen_strategy.generate_wait_for_health_function()

        srun_command = vllm_cmd_gen_strategy._gen_srun_command()

        expected = f"""\
cleanup() {{
    echo "Cleaning up PIDs: VLLM_PID=$VLLM_PID"
    [ -n "$VLLM_PID" ] && kill -9 $VLLM_PID 2>/dev/null
}}
trap cleanup EXIT

{health_func}

echo "Starting vLLM instances..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/{VLLM_SERVE_LOG_FILE} \\
    {serve_cmd} &
VLLM_PID=$!

NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
echo "Waiting for vLLM on $NODE to be ready..."
wait_for_health "http://${{NODE}}:{cmd_args.port}/health" || exit 1

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

        commands = strategy.get_vllm_serve_commands()

        assert len(commands) == 2
        prefill_cmd, decode_cmd = commands

        assert prefill_cmd == [
            "vllm",
            "serve",
            cmd_args.model,
            "--port",
            str(cmd_args.port + 100),
            "--kv-transfer-config",
            '\'{"kv_connector":"NixlConnector","kv_role":"kv_producer"}\'',
        ]
        assert decode_cmd == [
            "vllm",
            "serve",
            cmd_args.model,
            "--port",
            str(cmd_args.port + 200),
            "--kv-transfer-config",
            '\'{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}\'',
        ]

    def test_get_proxy_command(self, vllm_disagg_tr: TestRun, slurm_system: SlurmSystem) -> None:
        """Proxy command routes to prefill and decode ports."""
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_disagg_tr)
        cmd_args = vllm_disagg_tr.test.cmd_args

        command = strategy.get_proxy_command()

        assert command == [
            "python3",
            cmd_args.proxy_script,
            "--port",
            str(cmd_args.port),
            "--prefiller-hosts",
            "localhost",
            "--prefiller-ports",
            str(cmd_args.port + 100),
            "--decoder-hosts",
            "localhost",
            "--decoder-ports",
            str(cmd_args.port + 200),
        ]

    def test_gen_srun_command_disagg_flow(self, vllm_disagg_tr: TestRun, slurm_system: SlurmSystem) -> None:
        """Disagg mode starts prefill, decode, and proxy, waits for health checks."""
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_disagg_tr)
        cmd_args = vllm_disagg_tr.test.cmd_args
        output_path = vllm_disagg_tr.output_path.absolute()
        srun_prefix = " ".join(strategy.gen_srun_prefix())
        prefill_cmd, decode_cmd = strategy.get_vllm_serve_commands()
        proxy_cmd = strategy.get_proxy_command()
        bench_cmd = " ".join(strategy.get_vllm_bench_command())
        health_func = strategy.generate_wait_for_health_function()
        prefill_gpus = ",".join(str(g) for g in strategy.prefill_gpu_ids)
        decode_gpus = ",".join(str(g) for g in strategy.decode_gpu_ids)

        srun_command = strategy._gen_srun_command()

        expected = f"""\
cleanup() {{
    echo "Cleaning up PIDs: PREFILL_PID=$PREFILL_PID DECODE_PID=$DECODE_PID PROXY_PID=$PROXY_PID"
    [ -n "$PREFILL_PID" ] && kill -9 $PREFILL_PID 2>/dev/null
    [ -n "$DECODE_PID" ] && kill -9 $DECODE_PID 2>/dev/null
    [ -n "$PROXY_PID" ] && kill -9 $PROXY_PID 2>/dev/null
}}
trap cleanup EXIT

{health_func}

PORT_OFFSET=$((SLURM_JOB_ID % 1000))
PREFILL_NIXL_PORT=$((5557 + PORT_OFFSET))
DECODE_NIXL_PORT=$((5557 + PORT_OFFSET + 1))

echo "Starting vLLM instances..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --export=ALL,CUDA_VISIBLE_DEVICES={prefill_gpus},VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_NIXL_PORT \\
    --output={output_path}/vllm-prefill.log \\
    {" ".join(prefill_cmd)} &
PREFILL_PID=$!

{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --export=ALL,CUDA_VISIBLE_DEVICES={decode_gpus},VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_NIXL_PORT \\
    --output={output_path}/vllm-decode.log \\
    {" ".join(decode_cmd)} &
DECODE_PID=$!

NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
echo "Waiting for vLLM on $NODE to be ready..."
wait_for_health "http://${{NODE}}:{cmd_args.port + 100}/health" || exit 1
wait_for_health "http://${{NODE}}:{cmd_args.port + 200}/health" || exit 1

echo "Starting proxy..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/vllm-proxy.log \\
    {" ".join(proxy_cmd)} &
PROXY_PID=$!

echo "Running benchmark..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/{VLLM_BENCH_LOG_FILE} \\
    {bench_cmd}"""

        assert srun_command == expected
