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

from cloudai.workloads.common.llm_serving import LLMServingSlurmCommandGenStrategy, all_gpu_ids

from .sglang import (
    SGLANG_BENCH_JSONL_FILE,
    SGLANG_BENCH_LOG_FILE,
    SGLANG_SERVE_LOG_FILE,
    SglangArgs,
    SglangBenchCmdArgs,
    SglangTestDefinition,
)


def sglang_all_gpu_ids(tdef: SglangTestDefinition, system_gpus_per_node: int | None) -> list[int]:
    return all_gpu_ids(tdef, system_gpus_per_node)


class SglangSlurmCommandGenStrategy(LLMServingSlurmCommandGenStrategy[SglangTestDefinition]):
    """Command generation strategy for SGLang on Slurm systems."""

    @property
    def tdef(self) -> SglangTestDefinition:
        return cast(SglangTestDefinition, self.test_run.test)

    @property
    def healthcheck_path(self) -> str:
        endpoint = self.tdef.cmd_args.health_endpoint.strip()
        if not endpoint:
            return "/health"
        if endpoint.startswith("/"):
            return endpoint
        return f"/{endpoint}"

    @property
    def prefill_port(self) -> int:
        return self.tdef.cmd_args.port + 100

    @property
    def decode_port(self) -> int:
        return self.tdef.cmd_args.port + 200

    def get_sglang_serve_commands(self) -> list[list[str]]:
        cmd_args = self.tdef.cmd_args

        base_cmd = [
            "python3",
            "-m",
            cmd_args.serve_module,
            "--model-path",
            cmd_args.model,
            "--host",
            "0.0.0.0",
        ]
        if not cmd_args.prefill:
            return [[*base_cmd, "--port", str(cmd_args.port), *cmd_args.decode.serve_args]]

        commands: list[list[str]] = []
        for port, mode, args in [
            (self.prefill_port, "prefill", cast(SglangArgs, cmd_args.prefill)),
            (self.decode_port, "decode", cmd_args.decode),
        ]:
            backend = args.disaggregation_transfer_backend or "nixl"
            commands.append(
                [
                    *base_cmd,
                    "--port",
                    str(port),
                    "--disaggregation-mode",
                    mode,
                    "--disaggregation-transfer-backend",
                    str(backend),
                    *args.serve_args,
                ]
            )

        return commands

    def get_router_command(self) -> list[str]:
        return [
            "python3",
            "-m",
            self.tdef.cmd_args.router_module,
            "--pd-disaggregation",
            "--prefill",
            f"http://0.0.0.0:{self.prefill_port}",
            "--decode",
            f"http://0.0.0.0:{self.decode_port}",
            "--host",
            "0.0.0.0",
            "--port",
            str(self.tdef.cmd_args.port),
        ]

    def get_sglang_bench_command(self) -> list[str]:
        bench_args: SglangBenchCmdArgs = self.tdef.bench_cmd_args
        extra_args = bench_args.model_extra or {}
        extras = ["--" + key.replace("_", "-") + " " + str(value) for key, value in extra_args.items()]

        command = [
            "python3",
            "-m",
            self.tdef.cmd_args.bench_module,
            f"--backend {bench_args.backend}",
            f"--base-url http://0.0.0.0:{self.tdef.cmd_args.port}",
            f"--model {self.tdef.cmd_args.model}",
            f"--dataset-name {bench_args.dataset_name}",
            f"--num-prompts {bench_args.num_prompts}",
            f"--max-concurrency {bench_args.max_concurrency}",
            f"--random-input {bench_args.random_input}",
            f"--random-output {bench_args.random_output}",
            f"--warmup-requests {bench_args.warmup_requests}",
            f"--random-range-ratio {bench_args.random_range_ratio}",
            f"--output-file {(self.test_run.output_path / SGLANG_BENCH_JSONL_FILE).absolute()}",
            *extras,
        ]

        if bench_args.output_details:
            command.append("--output-details")

        if self.tdef.cmd_args.prefill:
            command.append("--pd-separated")

        return command

    def _gen_srun_command(self) -> str:
        srun_prefix = " ".join(self.gen_srun_prefix())
        serve_commands = self.get_sglang_serve_commands()
        bench_cmd = " ".join(self.get_sglang_bench_command())
        health_func = self.generate_wait_for_health_function()

        if len(serve_commands) == 1:
            return self._gen_aggregated_script(srun_prefix, serve_commands[0], bench_cmd, health_func)
        else:
            return self._gen_disaggregated_script(srun_prefix, serve_commands, bench_cmd, health_func)

    @staticmethod
    def _with_cuda_visible_devices(command: list[str], cuda_visible_devices: str) -> str:
        return " ".join(["env", f'CUDA_VISIBLE_DEVICES="{cuda_visible_devices}"', *command])

    def _gen_aggregated_script(self, srun_prefix: str, serve_cmd: list[str], bench_cmd: str, health_func: str) -> str:
        serve_gpus = ",".join(str(gpu_id) for gpu_id in self.gpu_ids)
        serve_cmd_with_env = self._with_cuda_visible_devices(serve_cmd, serve_gpus)
        return f"""\
cleanup() {{
    echo "Cleaning up PIDs: SGLANG_PID=$SGLANG_PID"
    [ -n "$SGLANG_PID" ] && kill -9 $SGLANG_PID 2>/dev/null
}}
trap cleanup EXIT

{health_func}

echo "Starting SGLang instances..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={(self.test_run.output_path / SGLANG_SERVE_LOG_FILE).absolute()} \\
    {serve_cmd_with_env} &
SGLANG_PID=$!

NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
echo "Waiting for SGLang on $NODE to be ready..."
wait_for_health "http://${{NODE}}:{self.tdef.cmd_args.port}{self.healthcheck_path}" || exit 1

echo "Running benchmark..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={(self.test_run.output_path / SGLANG_BENCH_LOG_FILE).absolute()} \\
    {bench_cmd}"""

    def _gen_disaggregated_script(
        self, srun_prefix: str, serve_commands: list[list[str]], bench_cmd: str, health_func: str
    ) -> str:
        prefill_cmd, decode_cmd = serve_commands
        router_cmd = self.get_router_command()
        prefill_gpus = ",".join(str(gpu_id) for gpu_id in self.prefill_gpu_ids)
        decode_gpus = ",".join(str(gpu_id) for gpu_id in self.decode_gpu_ids)
        prefill_cmd_with_env = self._with_cuda_visible_devices(prefill_cmd, prefill_gpus)
        decode_cmd_with_env = self._with_cuda_visible_devices(decode_cmd, decode_gpus)

        return f"""\
cleanup() {{
    echo "Cleaning up PIDs: PREFILL_PID=$PREFILL_PID DECODE_PID=$DECODE_PID ROUTER_PID=$ROUTER_PID"
    [ -n "$PREFILL_PID" ] && kill -9 $PREFILL_PID 2>/dev/null
    [ -n "$DECODE_PID" ] && kill -9 $DECODE_PID 2>/dev/null
    [ -n "$ROUTER_PID" ] && kill -9 $ROUTER_PID 2>/dev/null
}}
trap cleanup EXIT

{health_func}

echo "Starting SGLang instances..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={self.test_run.output_path.absolute()}/sglang-prefill.log \\
    {prefill_cmd_with_env} &
PREFILL_PID=$!

{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={self.test_run.output_path.absolute()}/sglang-decode.log \\
    {decode_cmd_with_env} &
DECODE_PID=$!

NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
echo "Waiting for SGLang on $NODE to be ready..."
wait_for_health "http://${{NODE}}:{self.prefill_port}{self.healthcheck_path}" || exit 1
wait_for_health "http://${{NODE}}:{self.decode_port}{self.healthcheck_path}" || exit 1

echo "Starting router..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={self.test_run.output_path.absolute()}/sglang-router.log \\
    {" ".join(router_cmd)} &
ROUTER_PID=$!

echo "Running benchmark..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={(self.test_run.output_path / SGLANG_BENCH_LOG_FILE).absolute()} \\
    {bench_cmd}"""
