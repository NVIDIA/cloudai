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

import json
from typing import Any, cast

from cloudai.workloads.common.llm_serving import LLMServingSlurmCommandGenStrategy, all_gpu_ids

from .vllm import (
    VLLM_BENCH_JSON_FILE,
    VLLM_BENCH_LOG_FILE,
    VLLM_SERVE_LOG_FILE,
    VllmCmdArgs,
    VllmTestDefinition,
)


def vllm_all_gpu_ids(tdef: VllmTestDefinition, system_gpus_per_node: int | None) -> list[int]:
    return all_gpu_ids(tdef, system_gpus_per_node)


class VllmSlurmCommandGenStrategy(LLMServingSlurmCommandGenStrategy[VllmCmdArgs]):
    """Command generation strategy for vLLM on Slurm systems."""

    @property
    def tdef(self) -> VllmTestDefinition:
        return cast(VllmTestDefinition, self.test_run.test)

    @staticmethod
    def _to_json_str_arg(config: dict) -> str:
        return "'" + json.dumps(config, separators=(",", ":")) + "'"

    def get_vllm_serve_commands(self) -> list[list[str]]:
        tdef: VllmTestDefinition = cast(VllmTestDefinition, self.test_run.test)
        cmd_args: VllmCmdArgs = tdef.cmd_args

        base_cmd = ["vllm", "serve", cmd_args.model]
        if not tdef.cmd_args.prefill:
            return [[*base_cmd, *tdef.cmd_args.decode.serve_args, "--port", str(cmd_args.port)]]

        prefill_port = cmd_args.port + 100
        decode_port = cmd_args.port + 200

        commands: list[list[str]] = []
        for port, role, args in [
            (prefill_port, "kv_producer", tdef.cmd_args.prefill),
            (decode_port, "kv_consumer", tdef.cmd_args.decode),
        ]:
            kv_transfer_config: dict[str, Any] = {"kv_connector": "NixlConnector", "kv_role": role}
            if args.nixl_threads is not None:
                kv_transfer_config["kv_connector_extra_config"] = {"num_threads": cast(int, args.nixl_threads)}
            commands.append(
                [
                    *base_cmd,
                    "--port",
                    str(port),
                    "--kv-transfer-config",
                    self._to_json_str_arg(kv_transfer_config),
                    *args.serve_args,
                ]
            )

        return commands

    def get_proxy_command(self) -> list[str]:
        prefill_port = self.tdef.cmd_args.port + 100
        decode_port = self.tdef.cmd_args.port + 200
        return [
            "python3",
            self.tdef.cmd_args.proxy_script,
            "--port",
            str(self.tdef.cmd_args.port),
            "--prefiller-hosts",
            "0.0.0.0",
            "--prefiller-ports",
            str(prefill_port),
            "--decoder-hosts",
            "0.0.0.0",
            "--decoder-ports",
            str(decode_port),
        ]

    def get_vllm_bench_command(self) -> list[str]:
        bench_args = self.tdef.bench_cmd_args
        extra_args = self.tdef.bench_cmd_args.model_extra or {}
        extras = ["--" + k.replace("_", "-") + " " + str(v) for k, v in extra_args.items()]
        return [
            "vllm",
            "bench",
            "serve",
            f"--model {self.tdef.cmd_args.model}",
            f"--base-url http://0.0.0.0:{self.tdef.cmd_args.port}",
            f"--random-input-len {bench_args.random_input_len}",
            f"--random-output-len {bench_args.random_output_len}",
            f"--max-concurrency {bench_args.max_concurrency}",
            f"--num-prompts {bench_args.num_prompts}",
            f"--result-dir {self.test_run.output_path.absolute()}",
            f"--result-filename {VLLM_BENCH_JSON_FILE}",
            "--save-result",
            *extras,
        ]

    def _gen_srun_command(self) -> str:
        serve_commands = self.get_vllm_serve_commands()
        bench_cmd = " ".join(self.get_vllm_bench_command())
        health_func = self.generate_wait_for_health_function()
        return self._gen_llm_serving_srun_command(serve_commands, bench_cmd, health_func)

    def _gen_aggregated_script(self, srun_prefix: str, serve_cmd: list[str], bench_cmd: str, health_func: str) -> str:
        return f"""\
{self.generate_cleanup_function(["VLLM_PID"])}

{health_func}

echo "Starting vLLM instances..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={(self.test_run.output_path / VLLM_SERVE_LOG_FILE).absolute()} \\
    {" ".join(serve_cmd)} &
VLLM_PID=$!

{self.generate_wait_for_health_block("vLLM", [f"http://${{NODE}}:{self.tdef.cmd_args.port}/health"])}

echo "Running benchmark..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={(self.test_run.output_path / VLLM_BENCH_LOG_FILE).absolute()} \\
    {bench_cmd}"""

    def _gen_disaggregated_script(
        self, srun_prefix: str, serve_commands: list[list[str]], bench_cmd: str, health_func: str
    ) -> str:
        prefill_cmd, decode_cmd = serve_commands
        proxy_cmd = self.get_proxy_command()
        prefill_port = self.tdef.cmd_args.port + 100
        decode_port = self.tdef.cmd_args.port + 200
        prefill_gpus = ",".join(str(g) for g in self.prefill_gpu_ids)
        decode_gpus = ",".join(str(g) for g in self.decode_gpu_ids)

        return f"""\
{self.generate_cleanup_function(["PREFILL_PID", "DECODE_PID", "PROXY_PID"])}

{health_func}

PORT_OFFSET=$((SLURM_JOB_ID % 1000))
PREFILL_NIXL_PORT=$((5557 + PORT_OFFSET))
DECODE_NIXL_PORT=$((5557 + PORT_OFFSET + {len(self.gpu_ids)}))

echo "Starting vLLM instances..."
export CUDA_VISIBLE_DEVICES="{prefill_gpus}"
export VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_NIXL_PORT
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={self.test_run.output_path.absolute()}/vllm-prefill.log \\
    {" ".join(prefill_cmd)} &
PREFILL_PID=$!

export CUDA_VISIBLE_DEVICES="{decode_gpus}"
export VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_NIXL_PORT
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={self.test_run.output_path.absolute()}/vllm-decode.log \\
    {" ".join(decode_cmd)} &
DECODE_PID=$!

{self.generate_wait_for_health_block(
    "vLLM",
    [
        f"http://${{NODE}}:{prefill_port}/health",
        f"http://${{NODE}}:{decode_port}/health",
    ],
)}

echo "Starting proxy..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={self.test_run.output_path.absolute()}/vllm-proxy.log \\
    {" ".join(proxy_cmd)} &
PROXY_PID=$!

echo "Running benchmark..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={(self.test_run.output_path / VLLM_BENCH_LOG_FILE).absolute()} \\
    {bench_cmd}"""
