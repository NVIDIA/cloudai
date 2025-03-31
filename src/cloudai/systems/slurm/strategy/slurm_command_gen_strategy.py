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

from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union, final

from cloudai import CommandGenStrategy, TestRun, TestScenario
from cloudai.systems import SlurmSystem


class SlurmCommandGenStrategy(CommandGenStrategy):
    """
    Abstract base class for defining command generation strategies specific to Slurm environments.

    Attributes
        system (SlurmSystem): A casted version of the `system` attribute, which provides Slurm-specific
            properties and methods.
    """

    def __init__(self, system: SlurmSystem, cmd_args: Dict[str, Any]) -> None:
        """
        Initialize a new SlurmCommandGenStrategy instance.

        Args:
            system (SlurmSystem): The system schema object.
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
            f"{self.system.install_path.absolute()}:/cloudai_install",
        ]

    def gen_exec_command(self, tr: TestRun) -> str:
        env_vars = self._override_env_vars(self.system.global_env_vars, tr.test.extra_env_vars)
        cmd_args = self._override_cmd_args(self.default_cmd_args, tr.test.cmd_args)
        slurm_args = self._parse_slurm_args(tr.test.test_template.__class__.__name__, env_vars, cmd_args, tr)

        srun_command = self._gen_srun_command(slurm_args, env_vars, cmd_args, tr)
        command_list = []
        indent = ""

        if tr.pre_test:
            pre_test_command = self.gen_pre_test(tr.pre_test, tr.output_path)
            command_list = [pre_test_command, "if [ $PRE_TEST_SUCCESS -eq 1 ]; then"]
            indent = "    "

        command_list.append(f"{indent}{srun_command}")

        if tr.post_test:
            post_test_command = self.gen_post_test(tr.post_test, tr.output_path)
            command_list.append(f"{indent}{post_test_command}")

        if tr.pre_test:
            command_list.append("fi")

        full_command = "\n".join(command_list).strip()
        return self._write_sbatch_script(slurm_args, env_vars, full_command, tr)

    def gen_srun_command(self, tr: TestRun) -> str:
        env_vars = self._override_env_vars(self.system.global_env_vars, tr.test.extra_env_vars)
        cmd_args = self._override_cmd_args(self.default_cmd_args, tr.test.cmd_args)
        slurm_args = self._parse_slurm_args(tr.test.test_template.__class__.__name__, env_vars, cmd_args, tr)
        return self._gen_srun_command(slurm_args, env_vars, cmd_args, tr)

    def _parse_slurm_args(
        self,
        job_name_prefix: str,
        env_vars: Dict[str, Union[str, List[str]]],
        cmd_args: Dict[str, Union[str, List[str]]],
        tr: TestRun,
    ) -> Dict[str, Any]:
        """
        Parse command arguments to configure Slurm job settings.

        Args:
            job_name_prefix (str): Prefix for the job name.
            env_vars (Dict[str, Union[str, List[str]]]): Environment variables.
            cmd_args (Dict[str, Union[str, List[str]]]): Command-line arguments.
            tr (TestRun): Test run object.

        Returns:
            Dict[str, Any]: Dictionary containing configuration for Slurm job.

        Raises:
            KeyError: If partition or essential node settings are missing.
        """
        job_name = self.job_name(job_name_prefix)
        num_nodes, node_list = self.system.get_nodes_by_spec(tr.num_nodes, tr.nodes)

        slurm_args = {
            "job_name": job_name,
            "num_nodes": num_nodes,
            "node_list_str": ",".join(node_list),
        }
        if tr.time_limit:
            slurm_args["time_limit"] = tr.time_limit

        return slurm_args

    def job_name(self, job_name_prefix: str) -> str:
        job_name = f"{job_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.system.account:
            job_name = f"{self.system.account}-{job_name_prefix}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return job_name

    def gen_pre_test(self, pre_test: TestScenario, base_output_path: Path) -> str:
        """
        Generate the pre-test command by running all tests defined in the pre-test test scenario.

        Args:
            pre_test (TestScenario): The pre-test test scenario containing the tests to be run.
            base_output_path (Path): The base output directory path for storing pre-test outputs.

        Returns:
            str: A string with all the Slurm srun commands generated for the pre_test.
        """
        pre_test_output_dir = base_output_path / "pre_test"
        pre_test_output_dir.mkdir(parents=True, exist_ok=True)

        pre_test_commands = []
        success_vars = []

        for idx, tr in enumerate(pre_test.test_runs):
            hook_dir = pre_test_output_dir / tr.test.name
            hook_dir.mkdir(parents=True, exist_ok=True)
            tr.output_path = hook_dir

            srun_command = tr.test.test_template.gen_srun_command(tr)
            srun_command_with_output = srun_command.replace(
                "srun ", f"srun --output={hook_dir / 'stdout.txt'} --error={hook_dir / 'stderr.txt'} "
            )
            pre_test_commands.append(srun_command_with_output)

            success_var = f"SUCCESS_{idx}"
            success_vars.append(success_var)

            success_check_command = tr.test.test_template.gen_srun_success_check(tr)
            pre_test_commands.append(f"{success_var}=$({success_check_command})")

        combined_success_var = " && ".join([f"[ ${var} -eq 1 ]" for var in success_vars])

        pre_test_commands.append(f"PRE_TEST_SUCCESS=$( {combined_success_var} && echo 1 || echo 0 )")

        return "\n".join(pre_test_commands)

    def gen_post_test(self, post_test: TestScenario, base_output_path: Path) -> str:
        """
        Generate the post-test command by running all tests defined in the post-test test scenario.

        Args:
            post_test (TestScenario): The post-test test scenario containing the tests to be run.
            base_output_path (Path): The base output directory path for storing post-test outputs.

        Returns:
            str: A string with all the Slurm srun commands generated for the post-test.
        """
        post_test_output_dir = base_output_path / "post_test"
        post_test_output_dir.mkdir(parents=True, exist_ok=True)

        post_test_commands = []

        for tr in post_test.test_runs:
            hook_dir = post_test_output_dir / tr.test.name
            hook_dir.mkdir(parents=True, exist_ok=True)
            tr.output_path = hook_dir

            srun_command = tr.test.test_template.gen_srun_command(tr)
            srun_command_with_output = srun_command.replace(
                "srun ", f"srun --output={hook_dir / 'stdout.txt'} --error={hook_dir / 'stderr.txt'} "
            )
            post_test_commands.append(srun_command_with_output)

        return "\n".join(post_test_commands)

    def gen_nsys_command(self, tr: TestRun) -> list[str]:
        nsys = tr.test.test_definition.nsys
        if not nsys or not nsys.enable:
            return []

        return nsys.cmd_args

    def _gen_srun_command(
        self,
        slurm_args: Dict[str, Any],
        env_vars: Dict[str, Union[str, List[str]]],
        cmd_args: Dict[str, Union[str, List[str]]],
        tr: TestRun,
    ) -> str:
        srun_command_parts = self.gen_srun_prefix(slurm_args, tr)
        nsys_command_parts = self.gen_nsys_command(tr)
        test_command_parts = self.generate_test_command(env_vars, cmd_args, tr)
        return " ".join(srun_command_parts + nsys_command_parts + test_command_parts)

    def gen_srun_prefix(self, slurm_args: Dict[str, Any], tr: TestRun) -> List[str]:
        srun_command_parts = ["srun", "--export=ALL", f"--mpi={self.system.mpi}"]
        if slurm_args.get("image_path"):
            srun_command_parts.append(f"--container-image={slurm_args['image_path']}")
            mounts = self.container_mounts(tr)
            if mounts:
                srun_command_parts.append(f"--container-mounts={','.join(mounts)}")

        if self.system.extra_srun_args:
            srun_command_parts.append(self.system.extra_srun_args)

        return srun_command_parts

    def generate_test_command(
        self, env_vars: Dict[str, Union[str, List[str]]], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        return []

    def _add_reservation(self, batch_script_content: List[str]):
        """
        Add reservation if provided.

        Args:
            batch_script_content (List[str]): content of the batch script.

        Returns:
            List[str]: updated batch script with reservation if exists.
        """
        reservation_key = "--reservation "
        if self.system.extra_srun_args and reservation_key in self.system.extra_srun_args:
            reservation = self.system.extra_srun_args.split(reservation_key, 1)[1].split(" ", 1)[0]
            batch_script_content.append(f"#SBATCH --reservation={reservation}")

        return batch_script_content

    def _ranks_mapping_cmd(self, slurm_args: dict[str, Any], tr: TestRun) -> str:
        return " ".join(
            [
                *self.gen_srun_prefix(slurm_args, tr),
                f"--output={tr.output_path.absolute() / 'mapping-stdout.txt'}",
                f"--error={tr.output_path.absolute() / 'mapping-stderr.txt'}",
                "bash",
                "-c",
                r'"echo \$(date): \$(hostname):node \${SLURM_NODEID}:rank \${SLURM_PROCID}."',
            ]
        )

    def _metadata_cmd(self, slurm_args: dict[str, Any], tr: TestRun) -> str:
        return " ".join(
            [
                *self.gen_srun_prefix(slurm_args, tr),
                "--ntasks-per-node=1",
                f"--output={tr.output_path.absolute() / 'metadata' / 'node-%N.toml'}",
                f"--error={tr.output_path.absolute() / 'metadata' / 'nodes.err'}",
                "bash",
                "/cloudai_install/slurm-metadata.sh",
            ]
        )

    def _write_sbatch_script(
        self, slurm_args: Dict[str, Any], env_vars: Dict[str, Union[str, List[str]]], srun_command: str, tr: TestRun
    ) -> str:
        """
        Write the batch script for Slurm submission and return the sbatch command.

        Args:
            slurm_args (Dict[str, Any]): Slurm-specific arguments.
            env_vars (Dict[str, Union[str, List[str]]]): Environment variables.
            srun_command (str): srun command.
            tr (TestRun): Test run object.

        Returns:
            str: sbatch command to submit the job.
        """
        batch_script_content = [
            "#!/bin/bash",
            f"#SBATCH --job-name={slurm_args['job_name']}",
            f"#SBATCH -N {slurm_args['num_nodes']}",
        ]

        self._append_sbatch_directives(batch_script_content, slurm_args, tr.output_path)

        batch_script_content.extend([self._format_env_vars(env_vars)])

        batch_script_content.extend([self._ranks_mapping_cmd(slurm_args, tr), ""])
        batch_script_content.extend([self._metadata_cmd(slurm_args, tr), ""])

        batch_script_content.append(srun_command)

        batch_script_path = tr.output_path / "cloudai_sbatch_script.sh"
        with batch_script_path.open("w") as batch_file:
            batch_file.write("\n".join(batch_script_content))

        return f"sbatch {batch_script_path}"

    def _append_sbatch_directives(
        self, batch_script_content: List[str], args: Dict[str, Any], output_path: Path
    ) -> None:
        """
        Append SBATCH directives to the batch script content.

        Args:
            batch_script_content (List[str]): The list of script lines to append to.
            args (Dict[str, Any]): Arguments including job settings.
            output_path (Path): Output directory for script and logs.
        """
        batch_script_content = self._add_reservation(batch_script_content)

        if "output" not in args:
            batch_script_content.append(f"#SBATCH --output={output_path / 'stdout.txt'}")
        if "error" not in args:
            batch_script_content.append(f"#SBATCH --error={output_path / 'stderr.txt'}")
        batch_script_content.append(f"#SBATCH --partition={self.system.default_partition}")
        if args["node_list_str"]:
            batch_script_content.append(f"#SBATCH --nodelist={args['node_list_str']}")
        if self.system.account:
            batch_script_content.append(f"#SBATCH --account={self.system.account}")
        if self.system.distribution:
            batch_script_content.append(f"#SBATCH --distribution={self.system.distribution}")
        if self.system.gpus_per_node:
            batch_script_content.append(f"#SBATCH --gpus-per-node={self.system.gpus_per_node}")
            batch_script_content.append(f"#SBATCH --gres=gpu:{self.system.gpus_per_node}")
        if self.system.ntasks_per_node:
            batch_script_content.append(f"#SBATCH --ntasks-per-node={self.system.ntasks_per_node}")
        if "time_limit" in args:
            batch_script_content.append(f"#SBATCH --time={args['time_limit']}")

        for arg in self.system.extra_sbatch_args:
            batch_script_content.append(f"#SBATCH {arg}")

        batch_script_content.append(
            "\nexport SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)"
        )

    def _format_env_vars(self, env_vars: Dict[str, Any]) -> str:
        """
        Format environment variables for inclusion in a batch script.

        Args:
            env_vars (Dict[str, Any]): Environment variables to format.

        Returns:
            str: A string representation of the formatted environment variables.
        """
        formatted_vars = []
        for key in sorted(env_vars.keys()):
            value = env_vars[key]
            formatted_value = str(value["default"]) if isinstance(value, dict) and "default" in value else str(value)
            formatted_vars.append(f"export {key}={formatted_value}")
        return "\n".join(formatted_vars)
