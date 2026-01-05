# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from abc import abstractmethod
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, cast, final

import toml

from cloudai.core import CommandGenStrategy, Registry, System, TestRun, TestScenario
from cloudai.models.scenario import TestRunDetails

from .slurm_system import SlurmSystem


class SlurmCommandGenStrategy(CommandGenStrategy):
    """
    Abstract base class for defining command generation strategies specific to Slurm environments.

    Attributes
        system (SlurmSystem): A casted version of the `system` attribute, which provides Slurm-specific
            properties and methods.
    """

    def __init__(self, system: System, test_run: TestRun) -> None:
        """
        Initialize a new SlurmCommandGenStrategy instance.

        Args:
            system (SlurmSystem): The system schema object.
            test_run (TestRun): The test run object.
        """
        super().__init__(system, test_run)
        self.system = cast(SlurmSystem, system)
        self.test_run = test_run

        self._node_spec_cache: dict[str, tuple[int, list[str]]] = {}

    @property
    def nodelist_in_use(self) -> bool:
        _, nodes = self.get_cached_nodes_spec()
        return len(nodes) > 0

    @abstractmethod
    def _container_mounts(self) -> list[str]:
        """Return CommandGenStrategy specific container mounts for the test run."""
        ...

    def store_test_run(self) -> None:
        test_cmd, srun_cmd = (" ".join(self.generate_test_command()), self.gen_srun_command())
        with (self.test_run.output_path / self.TEST_RUN_DUMP_FILE_NAME).open("w") as f:
            trd = TestRunDetails.from_test_run(self.test_run, test_cmd=test_cmd, full_cmd=srun_cmd)
            toml.dump(trd.model_dump(), f)

    @final
    def container_mounts(self) -> list[str]:
        """Return the container mounts for the test run."""
        tdef = self.test_run.test

        repo_mounts = []
        for repo in tdef.git_repos:
            path = repo.installed_path.absolute() if repo.installed_path else self.system.install_path / repo.repo_name
            repo_mounts.append(f"{path}:{repo.container_mount}")

        mounts = [
            f"{self.test_run.output_path.absolute()}:/cloudai_run_results",
            f"{self.system.install_path.absolute()}:/cloudai_install",
            f"{self.test_run.output_path.absolute()}",
            *tdef.extra_container_mounts,
            *repo_mounts,
            *self._container_mounts(),
        ]

        merged_env = self.system.global_env_vars.copy()
        merged_env.update(self.test_run.test.extra_env_vars)
        if "NCCL_TOPO_FILE" in merged_env:
            nccl_topo_file = merged_env["NCCL_TOPO_FILE"]
            if isinstance(nccl_topo_file, str):
                nccl_topo_file_path = Path(nccl_topo_file).resolve()
                mounts.append(f"{nccl_topo_file_path}:{nccl_topo_file_path}")

        return mounts

    def gen_exec_command(self) -> str:
        srun_command = self._gen_srun_command()
        command_list = []
        indent = ""

        if self.test_run.pre_test:
            pre_test_command = self.gen_pre_test(self.test_run.pre_test, self.test_run.output_path)
            command_list = [pre_test_command, "if [ $PRE_TEST_SUCCESS -eq 1 ]; then"]
            indent = "    "

        command_list.append(f"{indent}{srun_command}")

        if self.test_run.post_test:
            post_test_command = self.gen_post_test(self.test_run.post_test, self.test_run.output_path)
            command_list.append(f"{indent}{post_test_command}")

        if self.test_run.pre_test:
            command_list.append("fi")

        full_command = "\n".join(command_list).strip()
        return self._write_sbatch_script(full_command)

    def gen_srun_command(self) -> str:
        return self._gen_srun_command()

    def job_name_prefix(self) -> str:
        return self.test_run.test.name

    def job_name(self) -> str:
        job_name_prefix = self.job_name_prefix()
        job_name = f"{job_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.system.account:
            job_name = f"{self.system.account}-{job_name_prefix}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return job_name

    def _get_cmd_gen_strategy(self, tr: TestRun) -> "SlurmCommandGenStrategy":
        """
        Prepare a test run by creating its hook directory, setting its output_path, and retrieving CommandGenStrategy.

        Args:
            tr (TestRun): The test run to prepare.

        Returns:
            CommandGenStrategy: The strategy instance.
        """
        strategy_cls = Registry().get_command_gen_strategy(type(self.system), type(tr.test))
        strategy = cast(SlurmCommandGenStrategy, strategy_cls(self.system, tr))
        return strategy

    def _set_hook_output_path(self, tr: TestRun, base_output_path: Path) -> None:
        tr.output_path = base_output_path / tr.test.name
        tr.output_path.mkdir(parents=True, exist_ok=True)

    def pre_test_srun_extra_args(self, tr: TestRun) -> list[str]:
        """
        Return extra arguments from pre-test to actual test.

        Returns:
            list[str]: List of extra arguments for the pre-test srun command.
        """
        return []

    def gen_pre_test(self, pre_test: TestScenario, base_output_path: Path) -> str:
        """
        Generate the pre-test command by running all tests defined in the pre-test scenario.

        Args:
            pre_test (TestScenario): The pre-test scenario containing tests to run.
            base_output_path (Path): Base output directory for storing pre-test outputs.

        Returns:
            str: A string with all the Slurm srun commands generated for the pre-test.
        """
        pre_test_commands = []
        success_vars = []

        for idx, tr in enumerate(pre_test.test_runs):
            strategy = self._get_cmd_gen_strategy(tr)
            strategy._set_hook_output_path(tr, base_output_path / "pre_test")
            srun_command = strategy.gen_srun_command()
            srun_command_with_output = srun_command.replace(
                "srun ", f"srun --output={tr.output_path / 'stdout.txt'} --error={tr.output_path / 'stderr.txt'} "
            )
            pre_test_commands.append(srun_command_with_output)

            success_var = f"SUCCESS_{idx}"
            success_vars.append(success_var)
            success_check_command = strategy.gen_srun_success_check()
            pre_test_commands.append(f"{success_var}=$({success_check_command})")

        combined_success_var = " && ".join([f"[ ${var} -eq 1 ]" for var in success_vars])
        pre_test_commands.append(f"PRE_TEST_SUCCESS=$( {combined_success_var} && echo 1 || echo 0 )")

        return "\n".join(pre_test_commands)

    def gen_post_test(self, post_test: TestScenario, base_output_path: Path) -> str:
        """
        Generate the post-test command by running all tests defined in the post-test scenario.

        Args:
            post_test (TestScenario): The post-test scenario containing tests to run.
            base_output_path (Path): Base output directory for storing post-test outputs.

        Returns:
            str: A string with all the Slurm srun commands generated for the post-test.
        """
        post_test_commands = []
        for tr in post_test.test_runs:
            strategy = self._get_cmd_gen_strategy(tr)
            strategy._set_hook_output_path(tr, base_output_path / "post_test")
            srun_command = strategy.gen_srun_command()
            srun_command_with_output = srun_command.replace(
                "srun ", f"srun --output={tr.output_path / 'stdout.txt'} --error={tr.output_path / 'stderr.txt'} "
            )
            post_test_commands.append(srun_command_with_output)

        return "\n".join(post_test_commands)

    def gen_nsys_command(self) -> list[str]:
        nsys = self.test_run.test.nsys
        if not nsys or not nsys.enable:
            return []

        return nsys.cmd_args

    def _gen_srun_command(self) -> str:
        srun_command_parts = self.gen_srun_prefix(use_pretest_extras=True)
        nsys_command_parts = self.gen_nsys_command()
        test_command_parts = self.generate_test_command()

        with (self.test_run.output_path / "env_vars.sh").open("w") as f:
            for key, value in self.final_env_vars.items():
                f.write(f'export {key}="{value}"\n')

        full_test_cmd = (
            f'bash -c "source {(self.test_run.output_path / "env_vars.sh").absolute()}; '
            + " ".join(nsys_command_parts + test_command_parts)
            + '"'
        )

        return " ".join(srun_command_parts) + " " + full_test_cmd

    def image_path(self) -> Optional[str]:
        return None

    def gen_srun_prefix(self, use_pretest_extras: bool = False, with_num_nodes: bool = True) -> List[str]:
        num_nodes, _ = self.get_cached_nodes_spec()
        srun_command_parts = ["srun", "--export=ALL", f"--mpi={self.system.mpi}"]
        if with_num_nodes and not self.nodelist_in_use:
            srun_command_parts.append(f"-N{num_nodes}")
        if use_pretest_extras and self.test_run.pre_test:
            for pre_tr in self.test_run.pre_test.test_runs:
                srun_command_parts.extend(self._get_cmd_gen_strategy(pre_tr).pre_test_srun_extra_args(self.test_run))

        if image_path := self.image_path():
            srun_command_parts.append(f"--container-image={image_path}")
            mounts = self.container_mounts()
            if mounts:
                srun_command_parts.append(f"--container-mounts={','.join(mounts)}")

            if not self.system.container_mount_home:
                srun_command_parts.append("--no-container-mount-home")

        if self.system.extra_srun_args:
            srun_command_parts.append(self.system.extra_srun_args)
        if self.test_run.extra_srun_args:
            srun_command_parts.append(self.test_run.extra_srun_args)

        return srun_command_parts

    def generate_test_command(self) -> List[str]:
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

    def _ranks_mapping_cmd(self) -> str:
        return " ".join(
            [
                *self.gen_srun_prefix(),
                f"--output={self.test_run.output_path.absolute() / 'mapping-stdout.txt'}",
                f"--error={self.test_run.output_path.absolute() / 'mapping-stderr.txt'}",
                "bash",
                "-c",
                r'"echo \$(date): \$(hostname):node \${SLURM_NODEID}:rank \${SLURM_PROCID}."',
            ]
        )

    def _metadata_cmd(self) -> str:
        (self.test_run.output_path.absolute() / "metadata").mkdir(parents=True, exist_ok=True)
        num_nodes, _ = self.get_cached_nodes_spec()
        metadata_script_path = "/cloudai_install"
        if not self.image_path():
            metadata_script_path = str(self.system.install_path.absolute())
        return " ".join(
            [
                *self.gen_srun_prefix(),
                f"--ntasks={num_nodes}",
                "--ntasks-per-node=1",
                f"--output={self.test_run.output_path.absolute() / 'metadata' / 'node-%N.toml'}",
                f"--error={self.test_run.output_path.absolute() / 'metadata' / 'nodes.err'}",
                "bash",
                f"{metadata_script_path}/slurm-metadata.sh",
            ]
        )

    def _enable_vboost_cmd(self) -> str:
        num_nodes, _ = self.system.get_nodes_by_spec(self.test_run.nnodes, self.test_run.nodes)
        return " ".join(
            [
                "srun",
                f"--ntasks={num_nodes}",
                f"--output={self.test_run.output_path.absolute() / 'vboost.out'}",
                f"--error={self.test_run.output_path.absolute() / 'vboost.err'}",
                "bash",
                "-c",
                '"sudo nvidia-smi boost-slider --vboost 1"',
            ]
        )

    def _write_sbatch_script(self, srun_command: str) -> str:
        """
        Write the batch script for Slurm submission and return the sbatch command.

        Args:
            srun_command (str): srun command.

        Returns:
            str: sbatch command to submit the job.
        """
        batch_script_content = [
            "#!/bin/bash",
            f"# generated by CloudAI@{version('cloudai')}",
            f"#SBATCH --job-name={self.job_name()}",
        ]

        self._append_sbatch_directives(batch_script_content)

        batch_script_content.extend([self._format_env_vars(self.final_env_vars)])

        if self.final_env_vars.get("ENABLE_VBOOST") == "1":
            batch_script_content.extend([self._enable_vboost_cmd(), ""])
        batch_script_content.extend([self._ranks_mapping_cmd(), ""])
        batch_script_content.extend([self._metadata_cmd(), ""])

        batch_script_content.append(srun_command)

        batch_script_path = self.test_run.output_path / "cloudai_sbatch_script.sh"
        with batch_script_path.open("w") as batch_file:
            batch_file.write("\n".join(batch_script_content))

        return f"sbatch {batch_script_path}"

    def _append_sbatch_directives(self, batch_script_content: List[str]) -> None:
        """
        Append SBATCH directives to the batch script content.

        Args:
            batch_script_content (List[str]): The list of script lines to append to.
        """
        batch_script_content = self._add_reservation(batch_script_content)

        batch_script_content.append(f"#SBATCH --output={self.test_run.output_path.absolute() / 'stdout.txt'}")
        batch_script_content.append(f"#SBATCH --error={self.test_run.output_path.absolute() / 'stderr.txt'}")
        batch_script_content.append(f"#SBATCH --partition={self.system.default_partition}")
        if self.system.account:
            batch_script_content.append(f"#SBATCH --account={self.system.account}")

        hostfile = self._append_nodes_related_directives(batch_script_content)

        if self.system.gpus_per_node and self.system.supports_gpu_directives:
            batch_script_content.append(f"#SBATCH --gpus-per-node={self.system.gpus_per_node}")
            batch_script_content.append(f"#SBATCH --gres=gpu:{self.system.gpus_per_node}")

        if self.system.ntasks_per_node:
            batch_script_content.append(f"#SBATCH --ntasks-per-node={self.system.ntasks_per_node}")
        if self.test_run.time_limit:
            batch_script_content.append(f"#SBATCH --time={self.test_run.time_limit}")

        for arg in self.system.extra_sbatch_args:
            batch_script_content.append(f"#SBATCH {arg}")

        if hostfile is not None:
            batch_script_content.append(f"export SLURM_HOSTFILE={hostfile}")

        batch_script_content.append(
            "\nexport SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)"
        )

    def _append_nodes_related_directives(self, content: List[str]) -> Optional[Path]:
        num_nodes, node_list = self.get_cached_nodes_spec()

        if self.system.distribution:
            content.append(f"#SBATCH --distribution={self.system.distribution}")

        if node_list:
            content.append(f"#SBATCH --nodelist={','.join(node_list)}")

            hostfile = (self.test_run.output_path / "hostfile.txt").absolute()
            with hostfile.open("w") as hf:
                tasks = self.system.ntasks_per_node or 1
                for node in node_list:
                    for _ in range(tasks):
                        hf.write(f"{node}\n")

            return hostfile

        content.append(f"#SBATCH -N {num_nodes}")

        return None

    def _format_env_vars(self, env_vars: Dict[str, Any]) -> str:
        """
        Format environment variables for inclusion in a batch script.

        Args:
            env_vars (Dict[str, Any]): Environment variables to format.

        Returns:
            str: A string representation of the formatted environment variables.
        """
        formatted_vars = []
        for key, value in env_vars.items():
            formatted_value = str(value["default"]) if isinstance(value, dict) and "default" in value else str(value)
            formatted_vars.append(f"export {key}={formatted_value}")
        return "\n".join(formatted_vars)

    def gen_srun_success_check(self) -> str:
        """
        Generate the Slurm success check command to verify if a test run was successful.

        Returns:
            str: The generated command to check the success of the test run.
        """
        return ""

    def get_cached_nodes_spec(self) -> tuple[int, list[str]]:
        """
        Get nodes for a test run, using cache when available.

        It is needed to avoid multiple calls to the system.get_nodes_by_spec method which in turn queries the Slurm API.
        For a single test run it is not required, we can get actual nodes status only once.
        """
        cache_key = ":".join(
            [
                self.test_run.name,
                str(self.test_run.current_iteration),
                str(self.test_run.step),
                str(self.test_run.nnodes),
                ",".join(self.test_run.nodes),
            ]
        )

        if cache_key in self._node_spec_cache:
            logging.debug(f"Using cached node allocation for {cache_key}: {self._node_spec_cache[cache_key]}")
            return self._node_spec_cache[cache_key]

        self._node_spec_cache[cache_key] = self.system.get_nodes_by_spec(self.test_run.nnodes, self.test_run.nodes)
        return self._node_spec_cache[cache_key]
