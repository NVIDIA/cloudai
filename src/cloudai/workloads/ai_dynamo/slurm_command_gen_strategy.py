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
from textwrap import indent

import yaml

from cloudai.systems.slurm import SlurmCommandGenStrategy

from .ai_dynamo import AIDynamoTestDefinition, BaseModel


class AIDynamoSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for AI Dynamo on Slurm systems."""

    def _container_mounts(self) -> list[str]:
        td = cast(AIDynamoTestDefinition, self.test_run.test.test_definition)
        mounts = [
            f"{td.huggingface_home_host_path}:{td.cmd_args.huggingface_home_container_path}",
        ]
        mounts.append(f"{str(td.run_script.installed_path.absolute())}:/opt/run.sh")

        return mounts

    def image_path(self) -> str | None:
        tdef: AIDynamoTestDefinition = cast(AIDynamoTestDefinition, self.test_run.test.test_definition)
        return str(tdef.docker_image.installed_path)

    def _get_toml_args(self, base_model: BaseModel, prefix: str, exclude: List[str] = []) -> List[str]:
        args = []
        toml_args = base_model.model_dump()
        for k, v in toml_args.items():
            if k not in exclude:
                args.append(f'{prefix}{k} "{v}"')

        return args

    def _gen_script_args(self, td: AIDynamoTestDefinition) -> List[str]:
        args = []
        
        args.extend([
            f"--huggingface-home {td.cmd_args.huggingface_home_container_path}",
            f'--node-setup-cmd "{td.cmd_args.node_setup_cmd}"',
            "--results-dir /cloudai_run_results",
            f"--dynamo-frontend-node $SLURM_JOB_MASTER_NODE",
            f"--dynamo-num-prefill-nodes {td.cmd_args.dynamo.prefill_worker.num_nodes}",
            f"--dynamo-num-decode-nodes {td.cmd_args.dynamo.decode_worker.num_nodes}",
            f'--dynamo-extra-args-prefill "{td.cmd_args.dynamo.prefill_worker.extra_args}"',
            f'--dynamo-extra-args-decode "{td.cmd_args.dynamo.decode_worker.extra_args}"',
            f'--dynamo-extra-args-genai-perf "{td.cmd_args.genai_perf.extra_args}"',
        ])

        args.extend(self._get_toml_args(td.cmd_args.dynamo, "--dynamo-", exclude=["prefill_worker", "decode_worker", "genai_perf"]))
        args.extend(self._get_toml_args(td.cmd_args.dynamo.prefill_worker, "--prefill-", exclude=["num_nodes", "extra_args"]))
        args.extend(self._get_toml_args(td.cmd_args.dynamo.decode_worker, "--decode-", exclude=["num_nodes", "extra_args"]))
        args.extend(self._get_toml_args(td.cmd_args.genai_perf, "--genai-perf-", exclude=["extra_args"]))
        
        # Extra args from cmd_args
        if td.cmd_args.extra_args:
            # Split extra_args string and add them
            extra_args_list = td.cmd_args.extra_args.split()
            args.extend(extra_args_list)

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
                "bash",
                "/opt/run.sh",
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
