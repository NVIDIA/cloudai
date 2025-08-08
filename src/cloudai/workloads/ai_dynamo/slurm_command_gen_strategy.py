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
        mounts = [
            f"{td.huggingface_home_host_path}:{td.cmd_args.huggingface_home_container_path}",
            f"{td.script.installed_path.absolute()!s}:{td.script.installed_path.absolute()!s}",
        ]
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
            if k not in exclude:
                args.append(f'{prefix}{k} "{v}"')

        return args

    def _gen_script_args(self, td: AIDynamoTestDefinition) -> List[str]:
        args = []

        args.extend(
            [
                f"--huggingface-home {td.cmd_args.huggingface_home_container_path}",
                "--results-dir /cloudai_run_results",
            ]
        )

        args.extend(
            self._get_toml_args(
                td.cmd_args.dynamo, "--dynamo-", exclude=["prefill_worker", "decode_worker", "genai_perf"]
            )
        )
        args.extend(self._get_toml_args(td.cmd_args.dynamo.prefill_worker, "--prefill-"))
        args.extend(self._get_toml_args(td.cmd_args.dynamo.decode_worker, "--decode-"))
        args.extend(self._get_toml_args(td.cmd_args.genai_perf, "--genai-perf-"))

        return args

    def _gen_srun_command(self) -> str:
        td = cast(AIDynamoTestDefinition, self.test_run.test.test_definition)
        num_nodes, _ = self.get_cached_nodes_spec()
        srun_cmd = self.gen_srun_prefix()
        srun_cmd.extend(
            [
                f"--nodes={num_nodes}",
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

        assert isinstance(prefill_n, int), "prefill_worker.num_nodes must be an integer"
        assert isinstance(decode_n, int), "decode_worker.num_nodes must be an integer"

        total_nodes = prefill_n + decode_n

        requested_nodes, node_list = self.system.get_nodes_by_spec(self.test_run.nnodes, self.test_run.nodes)
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
