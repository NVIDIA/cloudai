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

from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union, final

from cloudai import CommandGenStrategy, TestRun, TestScenario
from cloudai.systems import LSFSystem


class LSFCommandGenStrategy(CommandGenStrategy):
    """
    Abstract base class for defining command generation strategies specific to LSF environments.

    Attributes:
        system (LSFSystem): A casted version of the `system` attribute, which provides LSF-specific
            properties and methods.
    """

    def __init__(self, system: LSFSystem, cmd_args: Dict[str, Any]) -> None:
        """
        Initialize a new LSFCommandGenStrategy instance.

        Args:
            system (LSFSystem): The system schema object.
            cmd_args (Dict[str, Any]): Command-line arguments.
        """
        super().__init__(system, cmd_args)
        self.system = system
        self.docker_image_url = self.cmd_args.get("docker_image_url", "")

    @abstractmethod
    def _container_mounts(self, tr: TestRun) -> list[str]:
        """Return CommandGenStrategy specific container mounts for the test run."""
        ...

    @final
    def container_mounts(self, tr: TestRun) -> list[str]:
        """
        Return the container mounts for the test run.

        Function returns CommandGenStrategy specific container mounts as well as default ones
        that should always be used.
        """
        tdef = tr.test.test_definition

        repo_mounts = []
        for repo in tdef.git_repos:
            path = repo.installed_path.absolute() if repo.installed_path else self.system.install_path / repo.repo_name
            repo_mounts.append(f"{path}:{repo.container_mount}")

        return [
            f"{tr.output_path.absolute()}:/cloudai_run_results",
            *tdef.extra_container_mounts,
            *repo_mounts,
            *self._container_mounts(tr),
        ]

    def gen_exec_command(self, tr: TestRun) -> str:
        """
        Generate the execution command for the test run.

        Args:
            tr (TestRun): The test run object.

        Returns:
            str: The generated LSF command.
        """
        env_vars = self._override_env_vars(self.system.global_env_vars, tr.test.extra_env_vars)
        cmd_args = self._override_cmd_args(self.default_cmd_args, tr.test.cmd_args)
        lsf_args = self._parse_lsf_args(tr.test.test_template.__class__.__name__, env_vars, cmd_args, tr)

        bsub_command = self._gen_bsub_command(lsf_args, env_vars, cmd_args, tr)
        
        return bsub_command.strip()

    def _parse_lsf_args(
        self,
        job_name_prefix: str,
        env_vars: Dict[str, Union[str, List[str]]],
        cmd_args: Dict[str, Union[str, List[str]]],
        tr: TestRun,
    ) -> Dict[str, Any]:
        """
        Parse command arguments to configure LSF job settings.

        Args:
            job_name_prefix (str): Prefix for the job name.
            env_vars (Dict[str, Union[str, List[str]]]): Environment variables.
            cmd_args (Dict[str, Union[str, List[str]]]): Command-line arguments.
            tr (TestRun): Test run object.

        Returns:
            Dict[str, Any]: Dictionary containing configuration for LSF job.
        """
        job_name = self.job_name(job_name_prefix)
        num_nodes = tr.num_nodes

        lsf_args = {
            "job_name": job_name,
            "num_nodes": num_nodes,
        }
        if tr.time_limit:
            lsf_args["time_limit"] = tr.time_limit

        return lsf_args

    def job_name(self, job_name_prefix: str) -> str:
        job_name = f"{job_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.system.account:
            job_name = f"{self.system.account}-{job_name_prefix}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return job_name

    def _gen_bsub_command(
        self,
        lsf_args: Dict[str, Any],
        env_vars: Dict[str, Union[str, List[str]]],
        cmd_args: Dict[str, Union[str, List[str]]],
        tr: TestRun,
    ) -> str:
        """
        Generate the LSF bsub command.

        Args:
            lsf_args (Dict[str, Any]): LSF-specific arguments.
            env_vars (Dict[str, Union[str, List[str]]]): Environment variables.
            cmd_args (Dict[str, Union[str, List[str]]]): Command-line arguments.
            tr (TestRun): The test run object.

        Returns:
            str: The generated bsub command.
        """
        bsub_command_parts = self.gen_bsub_prefix(lsf_args, tr)
        test_command_parts = self.generate_test_command(env_vars, cmd_args, tr)
        
        return " ".join(bsub_command_parts + test_command_parts)

    def gen_bsub_prefix(self, lsf_args: Dict[str, Any], tr: TestRun) -> List[str]:
        """
        Generate the prefix for the bsub command.

        Args:
            lsf_args (Dict[str, Any]): LSF-specific arguments.
            tr (TestRun): The test run object.

        Returns:
            List[str]: The prefix for the bsub command.
        """
        bsub_command_parts = ["qsub", f"-J {lsf_args['job_name']}"]

        if lsf_args.get("time_limit"):
            time_parts = list(map(int, lsf_args["time_limit"].split(":")))
            if len(time_parts) == 3: 
                time_limit_minutes = time_parts[0] * 60 + time_parts[1]
            elif len(time_parts) == 2: 
                time_limit_minutes = time_parts[0]
            else:  
                time_limit_minutes = 1
            bsub_command_parts.append(f"-W {time_limit_minutes}")

        if self.system.project_name:
            bsub_command_parts.append(f"-P {self.system.project_name}")

        if self.system.default_queue:
            bsub_command_parts.append(f"-q {self.system.default_queue}")

        if self.system.app:
            bsub_command_parts.append(f"-app {self.system.app}")
        if self.system.os_version:
            bsub_command_parts.append(f"-m {self.system.os_version}")

        return bsub_command_parts

    def generate_test_command(
        self, env_vars: Dict[str, Union[str, List[str]]], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        return []

    def gen_srun_command(self, tr: TestRun) -> str:
        """
        Generate the LSF bsub command for a test based on the given parameters.

        Args:
            tr (TestRun): Contains the test and its run-specific configurations.

        Returns:
            str: The generated LSF bsub command.
        """
        pass
