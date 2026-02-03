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


@pytest.fixture
def vllm() -> VllmTestDefinition:
    return VllmTestDefinition(
        name="vllm_test",
        description="vLLM benchmark test",
        test_template_name="Vllm",
        cmd_args=VllmCmdArgs(docker_image_url="nvcr.io/nvidia/vllm:latest", model="Qwen/Qwen3-0.6B", port=8000),
    )


@pytest.fixture
def vllm_tr(vllm: VllmTestDefinition, tmp_path: Path) -> TestRun:
    return TestRun(test=vllm, num_nodes=1, nodes=[], output_path=tmp_path, name="vllm-job")


@pytest.fixture
def vllm_cmd_gen_strategy(vllm_tr: TestRun, slurm_system: SlurmSystem) -> VllmSlurmCommandGenStrategy:
    return VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)


class TestVllmSlurmCommandGenStrategy:
    """Test the VllmSlurmCommandGenStrategy class."""

    def test_generate_vllm_command(self, vllm_tr: TestRun, slurm_system: SlurmSystem) -> None:
        cmd_gen_strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)

        command = " ".join(cmd_gen_strategy.get_vllm_serve_command())

        assert command == f"vllm serve {vllm_tr.test.cmd_args.model} --port {vllm_tr.test.cmd_args.port}"

    def test_generate_wait_for_health_function(self, vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy) -> None:
        """Test that wait_for_health bash function is generated correctly."""
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
        """Test that _gen_srun_command returns full flow: cleanup, server, health check, bench."""
        tdef = vllm_cmd_gen_strategy.test_run.test
        cmd_args = tdef.cmd_args
        output_path = vllm_cmd_gen_strategy.test_run.output_path.absolute()
        srun_prefix = " ".join(vllm_cmd_gen_strategy.gen_srun_prefix())
        serve_cmd = " ".join(vllm_cmd_gen_strategy.get_vllm_serve_command())
        bench_cmd = " ".join(vllm_cmd_gen_strategy.get_vllm_bench_command())
        health_func = vllm_cmd_gen_strategy.generate_wait_for_health_function()

        srun_command = vllm_cmd_gen_strategy._gen_srun_command()

        expected = f"""\
cleanup() {{
    [ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null
}}
trap cleanup EXIT

{health_func}

{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/vllm-serve-stdout.txt \\
    --error={output_path}/vllm-serve-stderr.txt \\
    {serve_cmd} &
VLLM_PID=$!

NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
wait_for_health "http://${{NODE}}:{cmd_args.port}/health" || exit 1

{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={output_path}/vllm-bench-stdout.txt \\
    --error={output_path}/vllm-bench-stderr.txt \\
    {bench_cmd}"""

        assert srun_command == expected
