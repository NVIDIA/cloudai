# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict, List, Union, cast

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy

from .nim import NimTestDefinition


class NimSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for NIM."""

    def _container_mounts(self, tr: TestRun) -> List[str]:
        return []

    def _parse_slurm_args(
        self,
        job_name_prefix: str,
        env_vars: Dict[str, Union[str, List[str]]],
        cmd_args: Dict[str, Union[str, List[str]]],
        tr: TestRun,
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

        tdef: NimTestDefinition = cast(NimTestDefinition, tr.test.test_definition)
        base_args.update({"image_path": tdef.docker_image.installed_path})

        return base_args

    def generate_test_command(
        self,
        env_vars: dict[str, Union[str, List[str]]],
        cmd_args: dict[str, Union[str, List[str]]],
        tr: TestRun,
    ) -> List[str]:
        test_definition = cast(NimTestDefinition, tr.test.test_definition)
        args = test_definition.cmd_args

        command = [
            "genai-perf",
            "profile",
            "-m",
            args.served_model_name,
            "--endpoint-type",
            args.endpoint_type,
            "--service-kind",
            args.service_kind,
        ]

        if args.streaming:
            command.append("--streaming")

        command += [
            "-u",
            f"{args.leader_ip}:{args.port}",
            "--num-prompts",
            str(args.num_prompts),
            "--synthetic-input-tokens-mean",
            str(args.input_sequence_length),
            "--synthetic-input-tokens-stddev",
            "0",
            "--concurrency",
            str(args.concurrency),
            "--output-tokens-mean",
            str(args.output_sequence_length),
            "--extra-inputs",
            f"max_tokens:{args.output_sequence_length}",
            "--extra-inputs",
            f"min_tokens:{args.output_sequence_length}",
            "--extra-inputs",
            "ignore_eos:true",
            "--artifact-dir",
            "/cloudai_run_results",
            "--tokenizer",
            args.tokenizer,
            "--",
            "-v",
            f"--max-threads {args.concurrency}",
            f"--request-count {args.num_prompts}",
        ]

        return command
