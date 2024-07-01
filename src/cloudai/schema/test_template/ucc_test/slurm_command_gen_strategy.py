#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict, List

from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy

from .slurm_install_strategy import UCCTestSlurmInstallStrategy
from .template import UCCTest


class UCCTestSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for UCC tests on Slurm systems."""

    def gen_exec_command(
        self,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_env_vars: Dict[str, str],
        extra_cmd_args: str,
        output_path: str,
        num_nodes: int,
        nodes: List[str],
    ) -> str:
        final_env_vars = self._override_env_vars(self.default_env_vars, env_vars)
        final_env_vars = self._override_env_vars(final_env_vars, extra_env_vars)
        final_cmd_args = self._override_cmd_args(self.default_cmd_args, cmd_args)
        env_vars_str = self._format_env_vars(final_env_vars)

        collective = final_cmd_args.get("collective")
        if not collective or collective not in UCCTest.SUPPORTED_COLLECTIVES:
            raise KeyError("Collective name not specified or unsupported.")

        slurm_args = self._parse_slurm_args(collective, final_env_vars, final_cmd_args, num_nodes, nodes)
        srun_command = self._generate_srun_command(slurm_args, final_env_vars, final_cmd_args, extra_cmd_args)
        return self._write_sbatch_script(slurm_args, env_vars_str, srun_command, output_path)

    def _parse_slurm_args(
        self,
        job_name_prefix: str,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        num_nodes: int,
        nodes: List[str],
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)

        base_args.update(
            {
                "image_path": self.docker_image_cache_manager.ensure_docker_image(
                    self.docker_image_url,
                    UCCTestSlurmInstallStrategy.SUBDIR_PATH,
                    UCCTestSlurmInstallStrategy.DOCKER_IMAGE_FILENAME,
                ).docker_image_path,
            }
        )

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
            f"--container-image={slurm_args['image_path']}",
            "/opt/hpcx/ucc/bin/ucc_perftest",
        ]

        # Add collective, minimum bytes, and maximum bytes options if available
        if "collective" in cmd_args:
            srun_command_parts.append(f"-c {cmd_args['collective']}")
        if "b" in cmd_args:
            srun_command_parts.append(f"-b {cmd_args['b']}")
        if "e" in cmd_args:
            srun_command_parts.append(f"-e {cmd_args['e']}")

        # Append fixed string options for memory type and additional flags
        srun_command_parts.append("-m cuda")
        srun_command_parts.append("-F")

        # Append any extra command-line arguments provided
        if extra_cmd_args:
            srun_command_parts.append(extra_cmd_args)

        return " \\\n".join(srun_command_parts)
