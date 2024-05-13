# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Any, Dict, List

from cloudai.schema.core import System
from cloudai.schema.core.strategy import CommandGenStrategy, StrategyRegistry
from cloudai.schema.system import SlurmSystem
from cloudai.schema.system.slurm.strategy import SlurmCommandGenStrategy

from .slurm_install_strategy import NcclMinerSlurmInstallStrategy
from .template import NcclMiner


@StrategyRegistry.strategy(CommandGenStrategy, [SlurmSystem], [NcclMiner])
class NcclMinerSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """
    Command generation strategy for NCCL miner on Slurm systems.
    """

    def __init__(
        self,
        system: System,
        env_vars: Dict[str, Any],
        cmd_args: Dict[str, Any],
    ) -> None:
        super().__init__(system, env_vars, cmd_args)

    def gen_exec_command(
        self,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_env_vars: Dict[str, str],
        extra_cmd_args: str,
        output_path: str,
        nodes: List[str],
    ) -> str:
        final_env_vars = self._override_env_vars(self.default_env_vars, env_vars)
        final_env_vars = self._override_env_vars(final_env_vars, extra_env_vars)
        final_cmd_args = self._override_cmd_args(self.default_cmd_args, cmd_args)
        final_cmd_args["output_path"] = output_path
        env_vars_str = self._format_env_vars(final_env_vars)

        slurm_args = self._parse_slurm_args("NcclMiner", final_env_vars, final_cmd_args, nodes)
        srun_command = self._generate_srun_command(slurm_args, final_env_vars, final_cmd_args, extra_cmd_args)
        return self._write_sbatch_script(slurm_args, env_vars_str, srun_command, output_path)

    def _parse_slurm_args(
        self,
        job_name_prefix: str,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        nodes: List[str],
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, nodes)

        image_path = os.path.join(
            self.install_path,
            NcclMinerSlurmInstallStrategy.SUBDIR_PATH,
            NcclMinerSlurmInstallStrategy.DOCKER_IMAGE_FILENAME,
        )

        output_path = cmd_args["output_path"]
        container_mounts = f"{output_path}:{output_path}"
        if "NCCL_TOPO_FILE" in env_vars and "DOCKER_NCCL_TOPO_FILE" in env_vars:
            nccl_graph_path = os.path.abspath(env_vars["NCCL_TOPO_FILE"])
            nccl_graph_file = env_vars["DOCKER_NCCL_TOPO_FILE"]
            container_mounts = f",{nccl_graph_path}:{nccl_graph_file}"
        elif "NCCL_TOPO_FILE" in env_vars:
            del env_vars["NCCL_TOPO_FILE"]

        base_args.update({"image_path": image_path, "container_mounts": container_mounts})

        return base_args

    def _generate_srun_command(
        self,
        slurm_args: Dict[str, Any],
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_cmd_args: str,
    ) -> str:
        srun_command_parts = [
            "srun",
            "--mpi=pmix",
            f'--container-image={slurm_args["image_path"]}',
            f'--container-mounts={slurm_args["container_mounts"]}',
            f'/app/container_run.sh {cmd_args["output_path"]}/nccl_miner_output.json',
        ]

        if extra_cmd_args:
            srun_command_parts.append(extra_cmd_args)

        return " \\\n".join(srun_command_parts)
