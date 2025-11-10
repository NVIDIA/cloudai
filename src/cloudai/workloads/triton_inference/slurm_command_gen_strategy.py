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

from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmCommandGenStrategy, SlurmSystem

from .triton_inference import TritonInferenceTestDefinition


class TritonInferenceSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for TritonInference server and client."""

    def __init__(self, system: SlurmSystem, test_run: TestRun) -> None:
        super().__init__(system, test_run)
        self._current_container_image: str | None = None

    def _container_mounts(self) -> list[str]:
        td = cast(TritonInferenceTestDefinition, self.test_run.test)
        mounts = [
            f"{td.nim_model_path}:{td.nim_model_path}:ro",
            f"{td.nim_cache_path}:{td.nim_cache_path}:rw",
        ]

        wrapper_host = (self.test_run.output_path / "start_server_wrapper.sh").resolve()
        wrapper_container = "/opt/nim/start_server_wrapper.sh"
        self._generate_start_wrapper_script(wrapper_host, td.extra_env_vars)
        mounts.append(f"{wrapper_host}:{wrapper_container}:ro")

        return mounts

    def _append_sbatch_directives(self, batch_script_content: List[str]) -> None:
        super()._append_sbatch_directives(batch_script_content)
        batch_script_content.append("export HEAD_NODE=$SLURM_JOB_MASTER_NODE")
        batch_script_content.append("export NIM_LEADER_IP_ADDRESS=$SLURM_JOB_MASTER_NODE")
        batch_script_content.append(f"export NIM_NUM_COMPUTE_NODES={self.test_run.nnodes - 1}")
        batch_script_content.append("export NIM_MODEL_TOKENIZER='deepseek-ai/DeepSeek-R1'")

    def _generate_start_wrapper_script(self, script_path: Path, env_vars: Dict[str, Any]) -> None:
        lines = ["#!/bin/bash", ""]
        lines.append("export NIM_LEADER_IP_ADDRESS=${SLURM_JOB_MASTER_NODE}")
        lines.append("export NIM_NODE_RANK=${SLURM_NODEID}")
        lines.append("")
        for key, val in env_vars.items():
            if key in {"NIM_LEADER_IP_ADDRESS", "NIM_NODE_RANK"}:
                continue
            if isinstance(val, str):
                lines.append(f"export {key}='{val}'")
        lines.append("")
        lines.append('if [ "$NIM_NODE_RANK" -eq 0 ]; then')
        lines.append("  export NIM_LEADER_ROLE=1")
        lines.append("else")
        lines.append("  export NIM_LEADER_ROLE=0")
        lines.append("fi")
        lines.append("")
        lines.append('echo "Starting NIM server on node rank ${NIM_NODE_RANK} with leader role ${NIM_LEADER_ROLE}"')
        lines.append("exec /opt/nim/start_server.sh")
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with script_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        script_path.chmod(0o755)

    def _gen_srun_command(self) -> str:
        num_server_nodes, num_client_nodes = self._get_server_client_split()
        server_line = self._build_server_srun(num_server_nodes)
        client_line = self._build_client_srun(num_client_nodes)
        sleep_sec = cast(TritonInferenceTestDefinition, self.test_run.test).cmd_args.sleep_seconds
        return f"{server_line} &\n\nsleep {sleep_sec}\n\n{client_line}"

    def _get_server_client_split(self) -> Tuple[int, int]:
        num_nodes, _ = self.system.get_nodes_by_spec(self.test_run.nnodes, self.test_run.nodes)
        if num_nodes < 3:
            raise ValueError("DeepSeekR1 requires at least 3 nodes: 2 server and 1 client.")
        return num_nodes - 1, 1

    def image_path(self) -> str | None:
        return self._current_container_image

    def _build_server_srun(self, num_server_nodes: int) -> str:
        test_definition = cast(TritonInferenceTestDefinition, self.test_run.test)
        self._current_container_image = str(test_definition.server_docker_image.installed_path)
        srun_prefix = self.gen_srun_prefix()
        self._current_container_image = None

        srun_prefix.append(f"--nodes={num_server_nodes}")
        srun_prefix.append(f"--ntasks={num_server_nodes}")
        srun_prefix.append("--ntasks-per-node=1")
        nsys_command = self.gen_nsys_command()
        server_launch_command = ["/opt/nim/start_server_wrapper.sh"]
        return " ".join(srun_prefix + nsys_command + server_launch_command)

    def _build_client_srun(self, num_client_nodes: int) -> str:
        test_definition = cast(TritonInferenceTestDefinition, self.test_run.test)
        self._current_container_image = str(test_definition.client_docker_image.installed_path)
        srun_prefix = self.gen_srun_prefix()
        self._current_container_image = None

        srun_prefix.append(f"--nodes={num_client_nodes}")
        srun_prefix.append(f"--ntasks={num_client_nodes}")

        args = test_definition.cmd_args
        client_command = [
            "genai-perf",
            "profile",
            "-m",
            args.served_model_name,
            f"--endpoint-type {args.endpoint_type}",
            f"--service-kind {args.service_kind}",
        ]
        if args.streaming:
            client_command.append("--streaming")
        client_command += [
            "-u",
            f"$SLURM_JOB_MASTER_NODE:{args.port}",
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
        return " ".join(srun_prefix + client_command)
