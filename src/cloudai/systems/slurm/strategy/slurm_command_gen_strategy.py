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
from abc import abstractmethod
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast, final

from cloudai import CommandGenStrategy, Registry, TestRun, TestScenario
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

        self._node_spec_cache: dict[str, tuple[int, list[str]]] = {}

    @abstractmethod
    def _container_mounts(self, tr: TestRun) -> list[str]:
        """Return CommandGenStrategy specific container mounts for the test run."""
        ...

    @final
    def container_mounts(self, tr: TestRun) -> list[str]:
        """Return the container mounts for the test run."""
        tdef = tr.test.test_definition

        repo_mounts = []
        for repo in tdef.git_repos:
            path = repo.installed_path.absolute() if repo.installed_path else self.system.install_path / repo.repo_name
            repo_mounts.append(f"{path}:{repo.container_mount}")

        mounts = [
            f"{tr.output_path.absolute()}:/cloudai_run_results",
            *tdef.extra_container_mounts,
            *repo_mounts,
            *self._container_mounts(tr),
            f"{self.system.install_path.absolute()}:/cloudai_install",
        ]

        merged_env = self.system.global_env_vars.copy()
        merged_env.update(tr.test.extra_env_vars)
        if "NCCL_TOPO_FILE" in merged_env:
            nccl_topo_file = merged_env["NCCL_TOPO_FILE"]
            if isinstance(nccl_topo_file, str):
                nccl_topo_file_path = Path(nccl_topo_file).resolve()
                mounts.append(f"{nccl_topo_file_path}:{nccl_topo_file_path}")

        return mounts

    def gen_exec_command(self, tr: TestRun) -> str:
        env_vars = self._override_env_vars(self.system.global_env_vars, tr.test.extra_env_vars)
        cmd_args = self._override_cmd_args(self.default_cmd_args, tr.test.cmd_args)

        srun_command = self._gen_srun_command(env_vars, cmd_args, tr)
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
        return self._write_sbatch_script(env_vars, full_command, tr)

    def gen_srun_command(self, tr: TestRun) -> str:
        env_vars = self._override_env_vars(self.system.global_env_vars, tr.test.extra_env_vars)
        cmd_args = self._override_cmd_args(self.default_cmd_args, tr.test.cmd_args)
        return self._gen_srun_command(env_vars, cmd_args, tr)

    def job_name_prefix(self, tr: TestRun) -> str:
        return tr.test.test_template.__class__.__name__

    def job_name(self, tr: TestRun) -> str:
        job_name_prefix = self.job_name_prefix(tr)
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
        registry = Registry()
        key = (CommandGenStrategy, type(self.system), type(tr.test.test_definition))
        strategy_cls = registry.strategies_map[key]
        strategy_cls_typed = cast(type[SlurmCommandGenStrategy], strategy_cls)
        strategy = strategy_cls_typed(self.system, tr.test.cmd_args)
        return strategy

    def _set_pre_test_output_path(self, tr: TestRun, base_output_path: Path) -> None:
        tr.output_path = base_output_path / "pre_test" / tr.test.name
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
            strategy._set_pre_test_output_path(tr, base_output_path)
            srun_command = strategy.gen_srun_command(tr)
            srun_command_with_output = srun_command.replace(
                "srun ", f"srun --output={tr.output_path / 'stdout.txt'} --error={tr.output_path / 'stderr.txt'} "
            )
            pre_test_commands.append(srun_command_with_output)

            success_var = f"SUCCESS_{idx}"
            success_vars.append(success_var)
            success_check_command = strategy.gen_srun_success_check(tr)
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
        post_test_output_dir = base_output_path / "post_test"
        post_test_output_dir.mkdir(parents=True, exist_ok=True)

        post_test_commands = []
        for tr in post_test.test_runs:
            strategy = self._get_cmd_gen_strategy(tr)
            strategy._set_pre_test_output_path(tr, post_test_output_dir)
            srun_command = strategy.gen_srun_command(tr)
            srun_command_with_output = srun_command.replace(
                "srun ", f"srun --output={tr.output_path / 'stdout.txt'} --error={tr.output_path / 'stderr.txt'} "
            )
            post_test_commands.append(srun_command_with_output)

        return "\n".join(post_test_commands)

    def gen_nsys_command(self, tr: TestRun) -> list[str]:
        nsys = tr.test.test_definition.nsys
        if not nsys or not nsys.enable:
            return []

        return nsys.cmd_args

    def _gen_srun_command(
        self, env_vars: Dict[str, Union[str, List[str]]], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> str:
        srun_command_parts = self.gen_srun_prefix(tr, use_pretest_extras=True)
        nsys_command_parts = self.gen_nsys_command(tr)
        test_command_parts = self.generate_test_command(env_vars, cmd_args, tr)
        return " ".join(srun_command_parts + nsys_command_parts + test_command_parts)

    def image_path(self, tr: TestRun) -> Optional[str]:
        return None

    def gen_srun_prefix(self, tr: TestRun, use_pretest_extras: bool = False) -> List[str]:
        srun_command_parts = ["srun", "--export=ALL", f"--mpi={self.system.mpi}"]
        if use_pretest_extras and tr.pre_test:
            for pre_tr in tr.pre_test.test_runs:
                srun_command_parts.extend(self._get_cmd_gen_strategy(pre_tr).pre_test_srun_extra_args(tr))

        if image_path := self.image_path(tr):
            srun_command_parts.append(f"--container-image={image_path}")
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

    def _ranks_mapping_cmd(self, tr: TestRun) -> str:
        return " ".join(
            [
                *self.gen_srun_prefix(tr),
                f"--output={tr.output_path.absolute() / 'mapping-stdout.txt'}",
                f"--error={tr.output_path.absolute() / 'mapping-stderr.txt'}",
                "bash",
                "-c",
                r'"echo \$(date): \$(hostname):node \${SLURM_NODEID}:rank \${SLURM_PROCID}."',
            ]
        )

    def _metadata_cmd(self, tr: TestRun) -> str:
        (tr.output_path.absolute() / "metadata").mkdir(parents=True, exist_ok=True)
        num_nodes, _ = self.get_cached_nodes_spec(tr)
        metadata_script_path = "/cloudai_install"
        if not self.image_path(tr):
            metadata_script_path = str(self.system.install_path.absolute())
        return " ".join(
            [
                *self.gen_srun_prefix(tr),
                f"--ntasks={num_nodes}",
                "--ntasks-per-node=1",
                f"--output={tr.output_path.absolute() / 'metadata' / 'node-%N.toml'}",
                f"--error={tr.output_path.absolute() / 'metadata' / 'nodes.err'}",
                "bash",
                f"{metadata_script_path}/slurm-metadata.sh",
            ]
        )

    def _enable_vboost_cmd(self, tr: TestRun) -> str:
        num_nodes, _ = self.system.get_nodes_by_spec(tr.num_nodes, tr.nodes)
        return " ".join(
            [
                "srun",
                f"--ntasks={num_nodes}",
                f"--output={tr.output_path.absolute() / 'vboost.out'}",
                f"--error={tr.output_path.absolute() / 'vboost.err'}",
                "bash",
                "-c",
                '"sudo nvidia-smi boost-slider --vboost 1"',
            ]
        )

    def _enable_numa_control_cmd(self, tr: TestRun) -> str:
        return " ".join(
            [
                "srun",
                f"--mpi={self.system.mpi}",
                "numactl",
                "--cpunodebind=$((SLURM_LOCALID/4))",
                "--membind=$((SLURM_LOCALID/4))",
            ]
        )

    def _write_sbatch_script(self, env_vars: Dict[str, Union[str, List[str]]], srun_command: str, tr: TestRun) -> str:
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
            f"# generated by CloudAI@{version('cloudai')}",
            f"#SBATCH --job-name={self.job_name(tr)}",
        ]

        self._append_sbatch_directives(batch_script_content, tr)

        batch_script_content.extend([self._format_env_vars(env_vars)])

        if env_vars.get("ENABLE_VBOOST") == "1":
            batch_script_content.extend([self._enable_vboost_cmd(tr), ""])
        if env_vars.get("ENABLE_NUMA_CONTROL") == "1":
            batch_script_content.extend([self._enable_numa_control_cmd(tr), ""])
        batch_script_content.extend([self._ranks_mapping_cmd(tr), ""])
        batch_script_content.extend([self._metadata_cmd(tr), ""])

        batch_script_content.append(srun_command)

        batch_script_path = tr.output_path / "cloudai_sbatch_script.sh"
        with batch_script_path.open("w") as batch_file:
            batch_file.write("\n".join(batch_script_content))

        return f"sbatch {batch_script_path}"

    def _append_sbatch_directives(self, batch_script_content: List[str], tr: TestRun) -> None:
        """
        Append SBATCH directives to the batch script content.

        Args:
            batch_script_content (List[str]): The list of script lines to append to.
            tr (TestRun): Test run object.
        """
        batch_script_content = self._add_reservation(batch_script_content)

        batch_script_content.append(f"#SBATCH --output={tr.output_path / 'stdout.txt'}")
        batch_script_content.append(f"#SBATCH --error={tr.output_path / 'stderr.txt'}")
        batch_script_content.append(f"#SBATCH --partition={self.system.default_partition}")
        if self.system.account:
            batch_script_content.append(f"#SBATCH --account={self.system.account}")

        hostfile = self._append_nodes_related_directives(batch_script_content, tr)

        if self.system.gpus_per_node:
            batch_script_content.append(f"#SBATCH --gpus-per-node={self.system.gpus_per_node}")
            batch_script_content.append(f"#SBATCH --gres=gpu:{self.system.gpus_per_node}")
        if self.system.ntasks_per_node:
            batch_script_content.append(f"#SBATCH --ntasks-per-node={self.system.ntasks_per_node}")
        if tr.time_limit:
            batch_script_content.append(f"#SBATCH --time={tr.time_limit}")

        for arg in self.system.extra_sbatch_args:
            batch_script_content.append(f"#SBATCH {arg}")

        if hostfile is not None:
            batch_script_content.append(f"export SLURM_HOSTFILE={hostfile}")

        batch_script_content.append(
            "\nexport SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)"
        )

    def _append_nodes_related_directives(self, content: List[str], tr: TestRun) -> Optional[Path]:
        num_nodes, node_list = self.get_cached_nodes_spec(tr)

        if node_list:
            content.append("#SBATCH --distribution=arbitrary")
            content.append(f"#SBATCH --nodelist={','.join(node_list)}")

            hostfile = (tr.output_path / "hostfile.txt").absolute()
            with hostfile.open("w") as hf:
                tasks = self.system.ntasks_per_node or 1
                for node in node_list:
                    for _ in range(tasks):
                        hf.write(f"{node}\n")

            return hostfile

        content.append(f"#SBATCH -N {num_nodes}")
        if self.system.distribution:
            content.append(f"#SBATCH --distribution={self.system.distribution}")

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
        for key in sorted(env_vars.keys()):
            value = env_vars[key]
            formatted_value = str(value["default"]) if isinstance(value, dict) and "default" in value else str(value)
            formatted_vars.append(f"export {key}={formatted_value}")
        return "\n".join(formatted_vars)

    def gen_srun_success_check(self, tr: TestRun) -> str:
        """
        Generate the Slurm success check command to verify if a test run was successful.

        Args:
            tr (TestRun): Contains the test and its run-specific configurations.

        Returns:
            str: The generated command to check the success of the test run.
        """
        return ""

    def get_cached_nodes_spec(self, tr: TestRun) -> tuple[int, list[str]]:
        """
        Get nodes for a test run, using cache when available.

        It is needed to avoid multiple calls to the system.get_nodes_by_spec method which in turn queries the Slurm API.
        For a single test run it is not required, we can get actual nodes status only once.
        """
        cache_key = f"{tr.current_iteration}:{tr.step}:{tr.num_nodes}:{','.join(tr.nodes)}"

        if cache_key in self._node_spec_cache:
            logging.debug(f"Using cached node allocation for {cache_key}: {self._node_spec_cache[cache_key]}")
            return self._node_spec_cache[cache_key]

        self._node_spec_cache[cache_key] = self.system.get_nodes_by_spec(tr.num_nodes, tr.nodes)
        return self._node_spec_cache[cache_key]
