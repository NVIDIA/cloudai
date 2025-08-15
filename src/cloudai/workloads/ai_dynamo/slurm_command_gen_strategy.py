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
from typing import List, cast

from cloudai.systems.slurm import SlurmCommandGenStrategy

from .ai_dynamo import AIDynamoTestDefinition, BaseModel


class AIDynamoSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for AI Dynamo on Slurm systems."""

    def _container_mounts(self) -> list[str]:
        td = cast(AIDynamoTestDefinition, self.test_run.test.test_definition)

        dynamo_repo_path = td.dynamo_repo.installed_path
        if dynamo_repo_path is None:
            raise ValueError("dynamo_repo_path is not set - repo may not be installed")
        dynamo_repo_path = dynamo_repo_path.absolute()

        mounts = [
            f"{dynamo_repo_path}:{dynamo_repo_path}",
            f"{td.cmd_args.huggingface_home_host_path}:{td.cmd_args.huggingface_home_container_path}",
            f"{td.script.installed_path.absolute()!s}:{td.script.installed_path.absolute()!s}",
        ]

        if td.cmd_args.dynamo.backend == "sglang":
            deepep_path = (
                dynamo_repo_path / "components/backends/sglang/configs/deepseek_r1/wideep/deepep.json"
            ).absolute()
            sgl_http_server_path = (
                dynamo_repo_path / "components/backends/sglang/src/dynamo/sglang/utils/sgl_http_server.py"
            ).absolute()
            mounts.extend(
                [
                    f"{deepep_path!s}:{deepep_path!s}",
                    f"{sgl_http_server_path!s}:{sgl_http_server_path!s}",
                ]
            )
        return mounts

    def image_path(self) -> str | None:
        tdef: AIDynamoTestDefinition = cast(AIDynamoTestDefinition, self.test_run.test.test_definition)
        if tdef.docker_image and tdef.docker_image.installed_path:
            return str(tdef.docker_image.installed_path)
        return None

    def _get_toml_args(self, base_model: BaseModel, prefix: str, exclude: List[str] | None = None) -> List[str]:
        args = []
        exclude = exclude or []
        toml_args = base_model.model_dump(by_alias=True)
        for k, v in toml_args.items():
            if k not in exclude and v is not None:
                args.append(f'{prefix}{k} "{v}"')

        return args

    def _gen_script_args(self, td: AIDynamoTestDefinition) -> List[str]:
        args = [
            f"--huggingface-home {td.cmd_args.huggingface_home_container_path}",
            "--results-dir /cloudai_run_results",
        ]
        args.extend(
            self._get_toml_args(
                td.cmd_args.dynamo, "--dynamo-", exclude=["prefill_worker", "decode_worker", "genai_perf"]
            )
        )

        # Add backend-specific args
        if td.cmd_args.dynamo.backend == "sglang":
            dynamo_repo_path = td.dynamo_repo.installed_path
            if dynamo_repo_path is None:
                raise ValueError("dynamo_repo_path is not set - repo may not be installed")

            deepep_path = getattr(td.cmd_args.dynamo, "deepep_path", None)
            if not deepep_path:
                deepep_path = (
                    dynamo_repo_path / "components/backends/sglang/configs/deepseek_r1/wideep/deepep.json"
                ).absolute()
            else:
                deepep_path = Path(deepep_path).absolute()

            sgl_http_server_path = (
                dynamo_repo_path / "components/backends/sglang/src/dynamo/sglang/utils/sgl_http_server.py"
            ).absolute()

            args.extend(
                [
                    f'--dynamo-sgl-http-server-script "{sgl_http_server_path!s}"',
                    f'--dynamo-deepep-config "{deepep_path!s}"',
                ]
            )
        elif td.cmd_args.dynamo.backend == "trtllm":
            prefill_engine_config = getattr(td.cmd_args.dynamo, "prefill_engine_config", None)
            decode_engine_config = getattr(td.cmd_args.dynamo, "decode_engine_config", None)

            if not prefill_engine_config:
                raise ValueError("prefill_engine_config is required for trtllm backend")
            if not decode_engine_config:
                raise ValueError("decode_engine_config is required for trtllm backend")

            prefill_engine_config = Path(prefill_engine_config).absolute()
            decode_engine_config = Path(decode_engine_config).absolute()

            args.extend(
                [
                    f'--dynamo-prefill-engine-config "{prefill_engine_config!s}"',
                    f'--dynamo-decode-engine-config "{decode_engine_config!s}"',
                ]
            )

        args.extend(self._get_toml_args(td.cmd_args.dynamo.prefill_worker, "--prefill-"))
        args.extend(self._get_toml_args(td.cmd_args.dynamo.decode_worker, "--decode-"))
        args.extend(self._get_toml_args(td.cmd_args.genai_perf, "--genai-perf-"))

        return args

    def _gen_srun_command(self) -> str:
        td = cast(AIDynamoTestDefinition, self.test_run.test.test_definition)
        num_nodes, node_list = self.get_cached_nodes_spec()
        srun_cmd = self.gen_srun_prefix()
        srun_cmd.extend(
            [
                f"--nodes={num_nodes}",
                *([] if not node_list else [f"--nodelist={','.join(node_list)}"]),
                f"--ntasks={num_nodes}",
                "--ntasks-per-node=1",
                f"--output={self.test_run.output_path.absolute() / 'node-%n-stdout.txt'}",
                f"--error={self.test_run.output_path.absolute() / 'node-%n-stderr.txt'}",
                "bash",
                f"{td.script.installed_path.absolute()!s}",
            ]
        )
        srun_cmd.extend(self._gen_script_args(td))
        return " \\\n  ".join(srun_cmd)

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

        td = cast(AIDynamoTestDefinition, self.test_run.test.test_definition)
        prefill_n = td.cmd_args.dynamo.prefill_worker.num_nodes
        decode_n = td.cmd_args.dynamo.decode_worker.num_nodes
        prefill_nodes = td.cmd_args.dynamo.prefill_worker.nodes
        decode_nodes = td.cmd_args.dynamo.decode_worker.nodes

        assert isinstance(prefill_n, int), "prefill_worker.num_nodes must be an integer"
        assert isinstance(decode_n, int), "decode_worker.num_nodes must be an integer"

        total_nodes = prefill_n + decode_n

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
