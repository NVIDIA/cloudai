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

from cloudai.workloads.common.llm_serving import LLMServingSlurmCommandGenStrategy

from .sglang import (
    SGLANG_BENCH_JSONL_FILE,
    SglangArgs,
    SglangBenchCmdArgs,
    SglangCmdArgs,
    SglangTestDefinition,
)


class SglangSlurmCommandGenStrategy(LLMServingSlurmCommandGenStrategy[SglangCmdArgs]):
    """Command generation strategy for SGLang on Slurm systems."""

    @property
    def tdef(self) -> SglangTestDefinition:
        return cast(SglangTestDefinition, self.test_run.test)

    @property
    def workload_name(self) -> str:
        return "SGLang"

    def get_serve_commands(self, prefill_host: str = "0.0.0.0", decode_host: str = "0.0.0.0") -> list[list[str]]:
        cmd_args = self.tdef.cmd_args

        base_cmd = ["python3", "-m", cmd_args.serve_module, "--model-path", cmd_args.model]
        if not cmd_args.prefill:
            return [
                [
                    *base_cmd,
                    "--host",
                    decode_host,
                    "--port",
                    str(self.serve_port),
                    *cmd_args.decode.serve_args,
                ]
            ]

        commands: list[list[str]] = []
        for host, port, mode, args in [
            (prefill_host, self.prefill_port, "prefill", cast(SglangArgs, cmd_args.prefill)),
            (decode_host, self.decode_port, "decode", cmd_args.decode),
        ]:
            commands.append(
                [
                    *base_cmd,
                    "--host",
                    host,
                    "--port",
                    str(port),
                    "--disaggregation-mode",
                    mode,
                    "--disaggregation-transfer-backend",
                    str(args.disaggregation_transfer_backend),
                    *args.serve_args,
                ]
            )

        return commands

    def get_helper_command(self) -> list[str]:
        return [
            "python3",
            "-m",
            self.tdef.cmd_args.router_module,
            "--pd-disaggregation",
            "--prefill",
            f"http://{self.disaggregated_role_host('prefill')}:{self.prefill_port}",
            "--decode",
            f"http://{self.disaggregated_role_host('decode')}:{self.decode_port}",
            "--host",
            "0.0.0.0",
            "--port",
            str(self.serve_port),
        ]

    def get_bench_command(self) -> list[str]:
        bench_args: SglangBenchCmdArgs = self.tdef.bench_cmd_args
        extra_args = bench_args.model_extra or {}
        extras = ["--" + key.replace("_", "-") + " " + str(value) for key, value in extra_args.items()]

        command = [
            "python3",
            "-m",
            self.tdef.cmd_args.bench_module,
            f"--backend {bench_args.backend}",
            f"--base-url http://127.0.0.1:{self.serve_port}",
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

    def aggregated_serve_env(self) -> dict[str, str]:
        return {"CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in self.gpu_ids)}
