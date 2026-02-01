# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from pathlib import Path, PosixPath
from typing import List, cast

from cloudai.systems.slurm import SlurmCommandGenStrategy

from .ai_dynamo import AIDynamoTestDefinition, BaseModel


class AIDynamoSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for AI Dynamo on Slurm systems."""

    def _container_mounts(self) -> list[str]:
        td = cast(AIDynamoTestDefinition, self.test_run.test)

        return [f"{self.system.hf_home_path.absolute()}:{td.cmd_args.dynamo.workspace_path}/hf_home"]

    def image_path(self) -> str | None:
        tdef: AIDynamoTestDefinition = cast(AIDynamoTestDefinition, self.test_run.test)
        if tdef.docker_image and tdef.docker_image.installed_path:
            return str(tdef.docker_image.installed_path)
        return None

    def _get_toml_args(self, base_model: BaseModel, prefix: str, exclude: List[str] | None = None) -> List[str]:
        args = []
        exclude = exclude or []
        toml_args = base_model.model_dump(by_alias=True, exclude=set(exclude), exclude_none=True)
        for k, v in toml_args.items():
            if isinstance(v, dict):
                if "url" in v and "commit" in v and "mount_as" in v:
                    args.extend([f'{prefix}{k} "{v["mount_as"]}"'])
                elif "src" in v and isinstance(v["src"], PosixPath):
                    args.extend([f'{prefix}{k} "{v["src"].name}"'])
                else:
                    args.append(f'{prefix}{k} "{v}"')
            else:
                args.append(f'{prefix}{k} "{v}"')

        return args

    def _get_nested_toml_args(self, base_model: BaseModel, prefix: str) -> List[str]:
        result = self._get_toml_args(base_model, prefix, exclude=["args"])

        if hasattr(base_model, "args") and (nested_args := getattr(base_model, "args", None)) is not None:
            result.extend(self._get_toml_args(nested_args, prefix + "args-"))

        return result

    def _gen_script_args(self, td: AIDynamoTestDefinition) -> List[str]:
        args = [
            "--user $USER",
            f"--install-dir {self.container_install_path}",
            f"--huggingface-home {td.cmd_args.dynamo.workspace_path}/hf_home",
            f"--results-dir {self.container_results_path}",
            f"--dynamo-repo {td.dynamo_repo.container_mount}",
        ]
        args.extend(
            self._get_toml_args(
                td.cmd_args.dynamo,
                "--dynamo-",
                exclude=[
                    "prefill_worker",
                    "decode_worker",
                ],
            )
        )

        # Add backend-specific args
        if td.cmd_args.dynamo.backend == "sglang":
            dynamo_repo_path = td.dynamo_repo.container_mount
            deepep_path = f"{dynamo_repo_path}/components/backends/sglang/configs/deepseek_r1/wideep/deepep.json"
            sgl_http_server_path = (
                f"{dynamo_repo_path}/components/backends/sglang/src/dynamo/sglang/utils/sgl_http_server.py"
            )

            args.extend(
                [
                    f'--dynamo-sgl-http-server-script "{sgl_http_server_path!s}"',
                    f'--dynamo-deepep-config "{deepep_path!s}"',
                ]
            )

        if td.cmd_args.dynamo.prefill_worker:
            args.extend(self._get_nested_toml_args(td.cmd_args.dynamo.prefill_worker, "--prefill-"))
        args.extend(self._get_nested_toml_args(td.cmd_args.dynamo.decode_worker, "--decode-"))

        args.extend(self._get_nested_toml_args(td.cmd_args.lmcache, "--lmcache-"))
        args.extend(self._get_nested_toml_args(td.cmd_args.genai_perf, "--genai_perf-"))
        args.extend(self._get_nested_toml_args(td.cmd_args.lmbench, "--lmbench-"))
        args.extend(self._get_nested_toml_args(td.cmd_args.custom_workload, "--custom_workload-"))

        return args

    def _gen_srun_command(self) -> str:
        td = cast(AIDynamoTestDefinition, self.test_run.test)
        num_nodes, node_list = self.get_cached_nodes_spec()

        out_dir = self.test_run.output_path.absolute()

        srun_cmd = self.gen_srun_prefix()
        srun_cmd.extend(
            [
                f"--nodes={num_nodes}",
                *([] if not node_list else [f"--nodelist={','.join(node_list)}"]),
                f"--ntasks={num_nodes}",
                "--ntasks-per-node=1",
                "--export=ALL",
                f"--output={out_dir / 'node-%n-stdout.txt'}",
                f"--error={out_dir / 'node-%n-stderr.txt'}",
                "bash",
                f"{self.container_install_path}/{td.script.src.name}",
            ]
        )
        srun_cmd.extend(self._gen_script_args(td))
        return " \\\n  ".join(srun_cmd) + "\n"

    def _validate_worker_nodes(
        self, node_list: list[str], worker_nodes: str | None, num_nodes: int, worker_type: str
    ) -> None:
        """Validate node list for a specific worker type."""
        if not worker_nodes:
            return

        worker_node_list = worker_nodes.split(",")
        if len(worker_node_list) != num_nodes:
            raise ValueError(
                f"Number of {worker_type} nodes ({len(worker_node_list)}) does not match num_nodes ({num_nodes})"
            )
        if not all(node in node_list for node in worker_node_list):
            raise ValueError(f"Some {worker_type} nodes are not in the allocated node list")

    def _validate_node_overlap(self, prefill_nodes: str, decode_nodes: str) -> None:
        """Validate that there is no overlap between prefill and decode nodes."""
        prefill_set = set(prefill_nodes.split(","))
        decode_set = set(decode_nodes.split(","))
        if prefill_set & decode_set:
            raise ValueError("Overlap found between prefill and decode node lists")

    def get_cached_nodes_spec(self) -> tuple[int, list[str]]:
        cache_key = ":".join(
            [
                self.test_run.name,
                str(self.test_run.current_iteration),
                str(self.test_run.step),
                str(self.test_run.num_nodes),
                ",".join(self.test_run.nodes),
            ]
        )

        if cache_key in self._node_spec_cache:
            return self._node_spec_cache[cache_key]

        td = cast(AIDynamoTestDefinition, self.test_run.test)
        prefill_n, prefill_nodes = 0, ""
        if td.cmd_args.dynamo.prefill_worker:
            prefill_n = cast(int, td.cmd_args.dynamo.prefill_worker.num_nodes)
            prefill_nodes = td.cmd_args.dynamo.prefill_worker.nodes
        decode_n = td.cmd_args.dynamo.decode_worker.num_nodes
        decode_nodes = td.cmd_args.dynamo.decode_worker.nodes

        assert isinstance(prefill_n, int), "dynamo.num_prefill_nodes must be an integer"
        assert isinstance(decode_n, int), "dynamo.num_decode_nodes must be an integer"

        if prefill_nodes and decode_nodes:
            self.test_run.nodes = prefill_nodes.split(",") + decode_nodes.split(",") + self.test_run.nodes
            self.test_run.num_nodes = len(self.test_run.nodes)
            prefill_n = len(prefill_nodes.split(","))
            decode_n = len(decode_nodes.split(","))
        else:
            self.test_run.num_nodes = prefill_n + decode_n

        total_nodes = prefill_n + decode_n

        logging.info("Setting num_nodes from %d to %d", self.test_run.num_nodes, total_nodes)

        self.test_run.num_nodes = total_nodes

        requested_nodes, node_list = self.system.get_nodes_by_spec(self.test_run.nnodes, self.test_run.nodes)

        if prefill_nodes or decode_nodes:
            self._validate_worker_nodes(node_list, prefill_nodes, prefill_n, "prefill")
            self._validate_worker_nodes(node_list, decode_nodes, decode_n, "decode")
            if prefill_nodes and decode_nodes:
                self._validate_node_overlap(prefill_nodes, decode_nodes)

        if total_nodes > requested_nodes:
            raise ValueError(
                f"Not enough nodes requested: need {total_nodes} total nodes "
                f"({prefill_n} prefill + {decode_n} decode), "
                f"but only got {requested_nodes}"
            )

        result = (total_nodes, node_list)
        self._node_spec_cache[cache_key] = result
        return result

    def gen_dynamo_cmd(self, module: str, config: Path) -> str:
        """
        Generate the dynamo command for serving a module with a config.

        Args:
            module: The module to serve.
            config: The path to the config file.

        Returns:
            The dynamo command string.
        """
        return f"dynamo serve {module} -f {config}"
