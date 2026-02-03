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

from typing import cast

from cloudai.systems.slurm import SlurmCommandGenStrategy

from .vllm import VLLM_BENCH_LOG_FILE, VLLM_SERVE_LOG_FILE, VllmCmdArgs, VllmTestDefinition


class VllmSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for vLLM on Slurm systems."""

    def _container_mounts(self) -> list[str]:
        return []

    def image_path(self) -> str | None:
        tdef: VllmTestDefinition = cast(VllmTestDefinition, self.test_run.test)
        return str(tdef.docker_image.installed_path)

    @property
    def gpu_ids(self) -> list[int]:
        cuda_devices = self.test_run.test.extra_env_vars.get("CUDA_VISIBLE_DEVICES")
        if cuda_devices:
            return [int(gpu_id) for gpu_id in str(cuda_devices).split(",")]
        return list(range(self.system.gpus_per_node or 1))

    @property
    def prefill_gpu_ids(self) -> list[int]:
        """Return first half of GPUs for prefill."""
        mid = len(self.gpu_ids) // 2
        return self.gpu_ids[:mid]

    @property
    def decode_gpu_ids(self) -> list[int]:
        """Return second half of GPUs for decode."""
        mid = len(self.gpu_ids) // 2
        return self.gpu_ids[mid:]

    def get_vllm_serve_commands(self) -> list[list[str]]:
        tdef: VllmTestDefinition = cast(VllmTestDefinition, self.test_run.test)
        cmd_args: VllmCmdArgs = tdef.cmd_args

        if len(self.gpu_ids) == 1:
            return [["vllm", "serve", cmd_args.model, "--port", str(cmd_args.port)]]

        prefill_port = cmd_args.port + 100
        decode_port = cmd_args.port + 200

        prefill_cmd = [
            "vllm",
            "serve",
            cmd_args.model,
            "--port",
            str(prefill_port),
            "--kv-transfer-config",
            '\'{"kv_connector":"NixlConnector","kv_role":"kv_producer"}\'',
        ]
        decode_cmd = [
            "vllm",
            "serve",
            cmd_args.model,
            "--port",
            str(decode_port),
            "--kv-transfer-config",
            '\'{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}\'',
        ]
        return [prefill_cmd, decode_cmd]

    def get_proxy_command(self) -> list[str]:
        """Return proxy server command for disaggregated mode."""
        tdef: VllmTestDefinition = cast(VllmTestDefinition, self.test_run.test)
        cmd_args: VllmCmdArgs = tdef.cmd_args
        prefill_port = cmd_args.port + 100
        decode_port = cmd_args.port + 200
        return [
            "python3",
            cmd_args.proxy_script,
            "--port",
            str(cmd_args.port),
            "--prefiller-hosts",
            "localhost",
            "--prefiller-ports",
            str(prefill_port),
            "--decoder-hosts",
            "localhost",
            "--decoder-ports",
            str(decode_port),
        ]

    def get_vllm_bench_command(self) -> list[str]:
        tdef: VllmTestDefinition = cast(VllmTestDefinition, self.test_run.test)
        cmd_args: VllmCmdArgs = tdef.cmd_args
        bench_args = tdef.bench_cmd_args
        return [
            "vllm",
            "bench",
            "serve",
            "--model",
            cmd_args.model,
            "--base-url",
            f"http://${{NODE}}:{cmd_args.port}",
            "--random-input-len",
            str(bench_args.random_input_len),
            "--random-output-len",
            str(bench_args.random_output_len),
            "--max-concurrency",
            str(bench_args.max_concurrency),
            "--num-prompts",
            str(bench_args.num_prompts),
        ]

    def generate_wait_for_health_function(self) -> str:
        """Generate bash function for health check."""
        tdef: VllmTestDefinition = cast(VllmTestDefinition, self.test_run.test)
        cmd_args: VllmCmdArgs = tdef.cmd_args

        return f"""\
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

    def _gen_srun_command(self) -> str:
        tdef: VllmTestDefinition = cast(VllmTestDefinition, self.test_run.test)
        cmd_args: VllmCmdArgs = tdef.cmd_args
        output_path = self.test_run.output_path.absolute()

        srun_prefix = " ".join(self.gen_srun_prefix())
        serve_commands = self.get_vllm_serve_commands()
        bench_cmd = " ".join(self.get_vllm_bench_command())
        health_func = self.generate_wait_for_health_function()

        if len(serve_commands) == 1:
            return self._gen_aggregated_script(
                srun_prefix, serve_commands[0], bench_cmd, health_func, cmd_args, output_path
            )
        else:
            return self._gen_disaggregated_script(
                srun_prefix, serve_commands, bench_cmd, health_func, cmd_args, output_path
            )

    def _gen_aggregated_script(
        self,
        srun_prefix: str,
        serve_cmd: list[str],
        bench_cmd: str,
        health_func: str,
        cmd_args: VllmCmdArgs,
        output_path,
    ) -> str:
        return f"""\
cleanup() {{
    [ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null
}}
trap cleanup EXIT

{health_func}

echo "Starting vLLM instances..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/{VLLM_SERVE_LOG_FILE} \\
    {" ".join(serve_cmd)} &
VLLM_PID=$!

NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
echo "Waiting for vLLM on $NODE to be ready..."
wait_for_health "http://${{NODE}}:{cmd_args.port}/health" || exit 1

echo "Running benchmark..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/{VLLM_BENCH_LOG_FILE} \\
    {bench_cmd}"""

    def _gen_disaggregated_script(
        self,
        srun_prefix: str,
        serve_commands: list[list[str]],
        bench_cmd: str,
        health_func: str,
        cmd_args: VllmCmdArgs,
        output_path,
    ) -> str:
        prefill_cmd, decode_cmd = serve_commands
        proxy_cmd = self.get_proxy_command()
        prefill_port = cmd_args.port + 100
        decode_port = cmd_args.port + 200
        prefill_gpus = ",".join(str(g) for g in self.prefill_gpu_ids)
        decode_gpus = ",".join(str(g) for g in self.decode_gpu_ids)

        return f"""\
cleanup() {{
    [ -n "$PREFILL_PID" ] && kill $PREFILL_PID 2>/dev/null
    [ -n "$DECODE_PID" ] && kill $DECODE_PID 2>/dev/null
    [ -n "$PROXY_PID" ] && kill $PROXY_PID 2>/dev/null
}}
trap cleanup EXIT

{health_func}

PORT_OFFSET=$((SLURM_JOB_ID % 1000))

echo "Starting vLLM instances..."
CUDA_VISIBLE_DEVICES={prefill_gpus} VLLM_NIXL_SIDE_CHANNEL_PORT=$((5557 + PORT_OFFSET)) \\
    {srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/vllm-prefill.log \\
    {" ".join(prefill_cmd)} &
PREFILL_PID=$!

CUDA_VISIBLE_DEVICES={decode_gpus} VLLM_NIXL_SIDE_CHANNEL_PORT=$((5557 + PORT_OFFSET + 1)) \\
    {srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/vllm-decode.log \\
    {" ".join(decode_cmd)} &
DECODE_PID=$!

NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
echo "Waiting for vLLM on $NODE to be ready..."
wait_for_health "http://${{NODE}}:{prefill_port}/health" || exit 1
wait_for_health "http://${{NODE}}:{decode_port}/health" || exit 1

echo "Starting proxy..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/vllm-proxy.log \\
    {" ".join(proxy_cmd)} &
PROXY_PID=$!

echo "Running benchmark..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/{VLLM_BENCH_LOG_FILE} \\
    {bench_cmd}"""
