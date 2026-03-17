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

from cloudai.workloads.common.llm_serving import LLMServingSlurmCommandGenStrategy

from .vllm import (
    VLLM_BENCH_JSON_FILE,
    VllmCmdArgs,
    VllmTestDefinition,
)


class VllmSlurmCommandGenStrategy(LLMServingSlurmCommandGenStrategy[VllmCmdArgs]):
    """Command generation strategy for vLLM on Slurm systems."""

    @property
    def tdef(self) -> VllmTestDefinition:
        return cast(VllmTestDefinition, self.test_run.test)

    @property
    def workload_name(self) -> str:
        return "vLLM"

    @staticmethod
    def _to_json_str_arg(config: dict) -> str:
        return "'" + json.dumps(config, separators=(",", ":")) + "'"

    def get_serve_commands(self) -> list[list[str]]:
        tdef: VllmTestDefinition = cast(VllmTestDefinition, self.test_run.test)
        cmd_args: VllmCmdArgs = tdef.cmd_args

        base_cmd = ["vllm", "serve", cmd_args.model]
        if not tdef.cmd_args.prefill:
            return [[*base_cmd, *tdef.cmd_args.decode.serve_args, "--port", str(self.serve_port)]]

        commands: list[list[str]] = []
        for port, role, args in [
            (self.prefill_port, "kv_producer", tdef.cmd_args.prefill),
            (self.decode_port, "kv_consumer", tdef.cmd_args.decode),
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

    def disaggregated_script_preamble(self) -> str:
        return f"""\
PORT_OFFSET=$((SLURM_JOB_ID % 1000))
PREFILL_NIXL_PORT=$((5557 + PORT_OFFSET))
DECODE_NIXL_PORT=$((5557 + PORT_OFFSET + {len(self.gpu_ids)}))

"""

    def disaggregated_role_env(self, role: str, gpu_ids: list[int]) -> dict[str, str]:
        env = super().disaggregated_role_env(role, gpu_ids)
        env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "$PREFILL_NIXL_PORT" if role == "prefill" else "$DECODE_NIXL_PORT"
        return env

    def get_helper_command(self, prefill_host: str, decode_host: str) -> list[str]:
        return [
            "python3",
            self.tdef.cmd_args.proxy_script,
            "--port",
            str(self.serve_port),
            "--prefiller-hosts",
            prefill_host,
            "--prefiller-ports",
            str(self.prefill_port),
            "--decoder-hosts",
            decode_host,
            "--decoder-ports",
            str(self.decode_port),
        ]

    def get_bench_command(self) -> list[str]:
        bench_args = self.tdef.bench_cmd_args
        extra_args = self.tdef.bench_cmd_args.model_extra or {}
        extras = ["--" + k.replace("_", "-") + " " + str(v) for k, v in extra_args.items()]
        return [
            "vllm",
            "bench",
            "serve",
            f"--model {self.tdef.cmd_args.model}",
            f"--base-url http://127.0.0.1:{self.serve_port}",
            f"--random-input-len {bench_args.random_input_len}",
            f"--random-output-len {bench_args.random_output_len}",
            f"--max-concurrency {bench_args.max_concurrency}",
            f"--num-prompts {bench_args.num_prompts}",
            f"--result-dir {self.test_run.output_path.absolute()}",
            f"--result-filename {VLLM_BENCH_JSON_FILE}",
            "--save-result",
            *extras,
        ]
