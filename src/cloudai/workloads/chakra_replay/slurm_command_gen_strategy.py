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

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union, cast

import toml

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
from cloudai.workloads.chakra_replay import ChakraReplayTestDefinition


class ChakraReplaySlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """ChakraReplaySlurmCommandGenStrategy."""

    def _container_mounts(self, tr: TestRun) -> List[str]:
        tdef = cast(ChakraReplayTestDefinition, tr.test.test_definition)
        trace_dir = tdef.cmd_args.trace_dir

        if not trace_dir:
            return []

        replay_exec = tdef.comm_replay_executable
        assert replay_exec.git_repo.installed_path is not None, "installed_path should never be None"

        installed_path: Path = replay_exec.git_repo.installed_path.resolve()

        mounts = [
            f"{trace_dir}:{trace_dir}",
            f"{installed_path}:{installed_path}",
        ]

        return [",".join(mounts)]

    def _parse_slurm_args(
        self, job_name_prefix: str, env_vars: Dict[str, str], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)
        tdef = cast(ChakraReplayTestDefinition, tr.test.test_definition)
        base_args.update({"image_path": str(tdef.docker_image.installed_path)})
        return base_args

    def _gen_srun_command(
        self,
        slurm_args: Dict[str, Any],
        env_vars: Dict[str, str],
        cmd_args: Dict[str, Union[str, List[str]]],
        tr: TestRun,
    ) -> str:
        config_parser = ChakraReplayConfigParser(cmd_args)
        config_path = config_parser.write_to_toml(tr.output_path).resolve()
        tdef = cast(ChakraReplayTestDefinition, tr.test.test_definition)
        assert tdef.comm_replay_executable.git_repo.installed_path is not None
        git_repo_path = tdef.comm_replay_executable.git_repo.installed_path.resolve()

        num_nodes = slurm_args.get("num_nodes", tr.num_nodes)
        ntasks_per_node = self.system.ntasks_per_node or 1
        total_tasks = num_nodes * ntasks_per_node

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        container_name = f"chakra_replay_container_{timestamp}"

        common_prefix = self.gen_srun_prefix(slurm_args, tr)

        install_prefix = [
            *common_prefix,
            f"--container-name={container_name}",
            f"-N {num_nodes}",
            f"-n {num_nodes}",
            "--ntasks-per-node=1",
        ]
        install_command = f'{" ".join(install_prefix)} bash -c "pip install {git_repo_path}"'

        run_prefix = [
            *common_prefix,
            f"--container-name={container_name}",
            f"-N {num_nodes}",
            f"-n {total_tasks}",
            f"--ntasks-per-node={ntasks_per_node}",
        ]
        run_command = f'{" ".join(run_prefix)} bash -c "comm_replay --config {config_path}"'

        return f"{install_command}\n{run_command}"


class ChakraReplayConfigParser:
    """ChakraReplayConfigParser."""

    def __init__(self, cmd_args: Dict[str, Union[str, List[str]]]) -> None:
        self.cmd_args = cmd_args
        self.config_data: Dict[str, Any] = {}
        self.parse()

    def parse(self) -> Dict[str, Any]:
        self._add_git_repo_config()
        self._add_run_config()
        self._add_trace_config()
        self._add_tensor_allocator_config()
        self._add_comm_config()
        self._add_profiler_config()
        self._add_logging_config()
        return self.config_data

    def _add_git_repo_config(self) -> None:
        if "git_repo.url" in self.cmd_args or "git_repo.commit" in self.cmd_args:
            self.config_data["git_repo"] = {
                key.split(".")[-1]: self.cmd_args[key]
                for key in ["git_repo.url", "git_repo.commit"]
                if key in self.cmd_args
            }

    def _add_run_config(self) -> None:
        if "warmup_iters" in self.cmd_args or "iters" in self.cmd_args:
            self.config_data["run"] = {
                key: self.cmd_args[key] for key in ["warmup_iters", "iters"] if key in self.cmd_args
            }

    def _add_trace_config(self) -> None:
        if "trace_dir" in self.cmd_args:
            self.config_data["trace"] = {"directory": self.cmd_args["trace_dir"]}

    def _add_tensor_allocator_config(self) -> None:
        if "reuse_tensors" in self.cmd_args:
            self.config_data["tensor_allocator"] = {"reuse_tensors": self.cmd_args["reuse_tensors"]}

    def _add_comm_config(self) -> None:
        if "backend.name" in self.cmd_args or "async_comm" in self.cmd_args:
            self.config_data["comm"] = {}

            if "backend.name" in self.cmd_args:
                self.config_data["comm"]["backend"] = {"name": self.cmd_args["backend.name"]}

            if "async_comm" in self.cmd_args:
                self.config_data["comm"]["async_comm"] = self.cmd_args["async_comm"]

    def _add_profiler_config(self) -> None:
        profiler_keys = [key for key in self.cmd_args if key.startswith("profiler.")]
        if profiler_keys:
            self.config_data["profiler"] = {key.split(".")[-1]: self.cmd_args[key] for key in profiler_keys}

    def _add_logging_config(self) -> None:
        logging_keys = [key for key in self.cmd_args if key.startswith("logging.")]
        if logging_keys:
            self.config_data["logging"] = {key.split(".")[-1]: self.cmd_args[key] for key in logging_keys}

    def write_to_toml(self, output_path: Path) -> Path:
        config_path = output_path / "config.toml"
        with config_path.open("w") as toml_file:
            toml.dump(self.config_data, toml_file)
        return config_path
