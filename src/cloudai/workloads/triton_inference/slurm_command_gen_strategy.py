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
from typing import Any, Dict, List, Tuple, Union, cast

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy

from .triton_inference import TritonInferenceTestDefinition


class TritonInferenceSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for TritonInference server and client."""

    def _container_mounts(self, tr: TestRun) -> List[str]:
        test_definition = cast(TritonInferenceTestDefinition, tr.test.test_definition)
        mounts: List[str] = []

        model_path_str = test_definition.extra_env_vars.get("NIM_MODEL_NAME")
        if not isinstance(model_path_str, str) or not model_path_str.strip():
            raise ValueError("NIM_MODEL_NAME must be set and non-empty.")
        model_path = Path(model_path_str)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found at: {model_path}")
        mounts.append(f"{model_path}:{model_path}:ro")

        cache_path_str = test_definition.extra_env_vars.get("NIM_CACHE_PATH")
        if not isinstance(cache_path_str, str) or not cache_path_str.strip():
            raise ValueError("NIM_CACHE_PATH must be set and non-empty.")
        cache_path = Path(cache_path_str)
        if not cache_path.is_dir():
            raise FileNotFoundError(f"NIM cache directory not found at: {cache_path}")
        mounts.append(f"{cache_path}:{cache_path}:rw")

        wrapper_host_path = (tr.output_path / "start_server_wrapper.sh").resolve()
        wrapper_container_path = "/opt/nim/start_server_wrapper.sh"
        self._generate_start_wrapper_script(wrapper_host_path, test_definition.extra_env_vars)
        mounts.append(f"{wrapper_host_path}:{wrapper_container_path}:ro")

        return mounts

    def _append_sbatch_directives(
        self,
        batch_script_content: List[str],
        args: Dict[str, Any],
        tr: TestRun,
    ) -> None:
        super()._append_sbatch_directives(batch_script_content, args, tr)
        batch_script_content.append("export HEAD_NODE=$SLURM_JOB_MASTER_NODE")
        batch_script_content.append("export NIM_LEADER_IP_ADDRESS=$SLURM_JOB_MASTER_NODE")
        batch_script_content.append(f"export NIM_NUM_COMPUTE_NODES={args['num_nodes'] - 1}")
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

    def _gen_srun_command(
        self,
        slurm_args: Dict[str, Any],
        env_vars: Dict[str, Union[str, List[str]]],
        cmd_args: Dict[str, Union[str, List[str]]],
        tr: TestRun,
    ) -> str:
        num_server_nodes, num_client_nodes = self._get_server_client_split(tr)
        server_line = self._build_server_srun(slurm_args, tr, num_server_nodes)
        client_line = self._build_client_srun(slurm_args, tr, num_client_nodes)
        sleep_sec = cast(TritonInferenceTestDefinition, tr.test.test_definition).cmd_args.sleep_seconds
        return f"{server_line} &\n\nsleep {sleep_sec}\n\n{client_line}"

    def _get_server_client_split(self, tr: TestRun) -> Tuple[int, int]:
        num_nodes, _ = self.system.get_nodes_by_spec(tr.num_nodes, tr.nodes)
        if num_nodes < 3:
            raise ValueError("DeepSeekR1 requires at least 3 nodes: 2 server and 1 client.")
        return num_nodes - 1, 1

    def _build_server_srun(self, slurm_args: Dict[str, Any], tr: TestRun, num_server_nodes: int) -> str:
        test_definition = cast(TritonInferenceTestDefinition, tr.test.test_definition)
        server_slurm_args = {
            **slurm_args,
            "image_path": test_definition.server_docker_image.installed_path,
        }
        srun_prefix = self.gen_srun_prefix(server_slurm_args, tr)
        srun_prefix.append(f"--nodes={num_server_nodes}")
        srun_prefix.append(f"--ntasks={num_server_nodes}")
        srun_prefix.append("--ntasks-per-node=1")
        nsys_command = self.gen_nsys_command(tr)
        server_launch_command = ["/opt/nim/start_server_wrapper.sh"]
        return " ".join(srun_prefix + nsys_command + server_launch_command)

    def _build_client_srun(self, slurm_args: Dict[str, Any], tr: TestRun, num_client_nodes: int) -> str:
        test_definition = cast(TritonInferenceTestDefinition, tr.test.test_definition)
        client_slurm_args = {
            **slurm_args,
            "image_path": test_definition.client_docker_image.installed_path,
        }
        srun_prefix = self.gen_srun_prefix(client_slurm_args, tr)
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
