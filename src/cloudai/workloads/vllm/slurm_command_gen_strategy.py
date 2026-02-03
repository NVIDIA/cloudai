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

from .vllm import VllmCmdArgs, VllmTestDefinition


class VllmSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for vLLM on Slurm systems."""

    def _container_mounts(self) -> list[str]:
        return []

    def image_path(self) -> str | None:
        tdef: VllmTestDefinition = cast(VllmTestDefinition, self.test_run.test)
        return tdef.cmd_args.docker_image_url

    def get_vllm_serve_command(self) -> list[str]:
        tdef: VllmTestDefinition = cast(VllmTestDefinition, self.test_run.test)
        tdef_cmd_args: VllmCmdArgs = tdef.cmd_args
        return ["vllm", "serve", tdef_cmd_args.model, "--port", str(tdef_cmd_args.port)]

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

        srun_prefix = " ".join(self.gen_srun_prefix())
        serve_cmd = " ".join(self.get_vllm_serve_command())
        bench_cmd = " ".join(self.get_vllm_bench_command())
        health_func = self.generate_wait_for_health_function()

        return f"""\
cleanup() {{
    [ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null
}}
trap cleanup EXIT

{health_func}

# Start vLLM in background
{srun_prefix} --ntasks-per-node=1 --ntasks=1 {serve_cmd} &
VLLM_PID=$!

# Wait for instances to be ready
NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
wait_for_health "http://${{NODE}}:{cmd_args.port}/health" || exit 1

# Run benchmark
{srun_prefix} --ntasks-per-node=1 --ntasks=1 {bench_cmd}"""
