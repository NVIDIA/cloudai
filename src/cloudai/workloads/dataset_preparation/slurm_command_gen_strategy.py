# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from datetime import datetime
from pathlib import Path
from typing import List, cast

from cloudai import TestRun
from cloudai.workloads.dataset_preparation import DatasetPreparationTestDefinition

from ..nemo_launcher import NeMoLauncherSlurmCommandGenStrategy


class DatasetPreparationSlurmCommandGenStrategy(NeMoLauncherSlurmCommandGenStrategy):
    """Command generation strategy for NeMo Dataset Preparation on Slurm."""

    job_prefix_name = "cloudai.dataprep"

    def gen_exec_command(self, tr: TestRun) -> str:
        tdef: DatasetPreparationTestDefinition = cast(DatasetPreparationTestDefinition, tr.test.test_definition)
        super()._prepare_environment(tr.test.cmd_args, tr.test.extra_env_vars, tr.output_path)

        self._set_job_name(tr)
        self._set_container_args(tdef)
        self._set_node_config(tr.nodes, tr.num_nodes)
        self.final_cmd_args.update(
            {
                "base_results_dir": str(tr.output_path.absolute()),
                "name": tr.test.name,
                "launcher_scripts_path": "${launcher_scripts_path}",
            }
        )

        py_bin, repo_path = self._get_paths(tdef)
        cmd_args_str = self._generate_cmd_args_str(self.final_cmd_args, tr.nodes)
        full_cmd = f"{py_bin} {repo_path / tdef.cmd_args.launcher_script} {cmd_args_str}"

        env_vars_str = " ".join(f"{key}={value}" for key, value in self.final_env_vars.items())
        full_cmd = f"{env_vars_str} {full_cmd}" if env_vars_str else full_cmd

        self._log_command_to_file(full_cmd, tr.output_path)
        return full_cmd.strip()

    def _set_job_name(self, tr: TestRun):
        if self.system.account:
            self.final_cmd_args["cluster.account"] = self.system.account

        if self.job_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.job_prefix = (
                f"{self.system.account}-{self.job_prefix_name}_{timestamp}" if self.system.account else timestamp
            )
        self.final_cmd_args["cluster.job_name_prefix"] = f"{self.job_prefix}:"

    def _set_container_args(self, tdef: DatasetPreparationTestDefinition):
        if tdef.docker_image:
            self.final_cmd_args["container"] = str(tdef.docker_image.installed_path)
            self.final_cmd_args.pop("docker_image_url", None)

    def _get_paths(self, tdef: DatasetPreparationTestDefinition) -> tuple[Path, Path]:
        repo_path = (
            tdef.python_executable.git_repo.installed_path.absolute()
            if tdef.python_executable.git_repo.installed_path is not None
            else None
        )
        if not repo_path:
            logging.warning(
                f"Local clone of git repo {tdef.python_executable.git_repo} does not exist. "
                "Please ensure to run installation before running the test."
            )
            repo_path = self.system.install_path / tdef.python_executable.git_repo.repo_name
        venv_path = tdef.python_executable.venv_path
        if not venv_path:
            logging.warning(
                f"The virtual environment for git repo {tdef.python_executable.git_repo} does not exist. "
                "Please ensure to run installation before running the test."
            )
            venv_path = self.system.install_path / tdef.python_executable.venv_name
        py_bin = (venv_path / "bin" / "python").absolute()

        self.final_cmd_args["launcher_scripts_path"] = str(
            (repo_path / tdef.cmd_args.launcher_script).parent.absolute()
        )
        return py_bin, repo_path

    def _set_node_config(self, nodes: List[str], num_nodes: int) -> None:
        if nodes:
            self.final_cmd_args["+cluster.nodelist"] = ",".join(nodes)
        elif num_nodes > 0:
            self.final_cmd_args["data_preparation.run.node_array_size"] = num_nodes
