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

from typing import Any, Dict, List, Union, cast

import toml

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
from cloudai.workloads.chakra_replay import ChakraReplayTestDefinition


class ChakraReplaySlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for ChakraReplay on Slurm systems."""

    def _container_mounts(self, tr: TestRun) -> list[str]:
        tdef: ChakraReplayTestDefinition = cast(ChakraReplayTestDefinition, tr.test.test_definition)
        if tdef.cmd_args.trace_dir:
            return [f"{tdef.cmd_args.trace_dir}:{tdef.cmd_args.trace_dir}"]
        return []

    def _parse_slurm_args(
        self, job_name_prefix: str, env_vars: Dict[str, str], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

        tdef: ChakraReplayTestDefinition = cast(ChakraReplayTestDefinition, tr.test.test_definition)
        base_args.update({"image_path": tdef.docker_image.installed_path})

        return base_args

    def _filter_config_data(self, cmd_args: Dict[str, Union[str, List[str]]]) -> Dict[str, Any]:
        config_data = {}

        self._add_git_repo_config(config_data, cmd_args)
        self._add_run_config(config_data, cmd_args)
        self._add_trace_config(config_data, cmd_args)
        self._add_tensor_allocator_config(config_data, cmd_args)
        self._add_comm_config(config_data, cmd_args)
        self._add_profiler_config(config_data, cmd_args)
        self._add_logging_config(config_data, cmd_args)

        return config_data

    def _add_git_repo_config(self, config_data: Dict[str, Any], cmd_args: Dict[str, Any]) -> None:
        if "git_repo.url" in cmd_args or "git_repo.commit" in cmd_args:
            config_data["git_repo"] = {}
            if "git_repo.url" in cmd_args:
                config_data["git_repo"]["url"] = cmd_args["git_repo.url"]
            if "git_repo.commit" in cmd_args:
                config_data["git_repo"]["commit"] = cmd_args["git_repo.commit"]

    def _add_run_config(self, config_data: Dict[str, Any], cmd_args: Dict[str, Any]) -> None:
        if "warmup_iters" in cmd_args or "iters" in cmd_args:
            config_data["run"] = {}
            if "warmup_iters" in cmd_args:
                config_data["run"]["warmup_iters"] = cmd_args["warmup_iters"]
            if "iters" in cmd_args:
                config_data["run"]["iters"] = cmd_args["iters"]

    def _add_trace_config(self, config_data: Dict[str, Any], cmd_args: Dict[str, Any]) -> None:
        if "trace_dir" in cmd_args:
            config_data["trace"] = {"directory": cmd_args["trace_dir"]}

    def _add_tensor_allocator_config(self, config_data: Dict[str, Any], cmd_args: Dict[str, Any]) -> None:
        if "reuse_tensors" in cmd_args:
            config_data["tensor_allocator"] = {"reuse_tensors": cmd_args["reuse_tensors"]}

    def _add_comm_config(self, config_data: Dict[str, Any], cmd_args: Dict[str, Any]) -> None:
        if "backend.name" in cmd_args or "async_comm" in cmd_args:
            config_data["comm"] = {}
            if "backend.name" in cmd_args:
                config_data["comm"]["backend"] = {"name": cmd_args["backend.name"]}
            if "async_comm" in cmd_args:
                config_data["comm"]["async_comm"] = cmd_args["async_comm"]

    def _add_profiler_config(self, config_data: Dict[str, Any], cmd_args: Dict[str, Any]) -> None:
        profiler_keys = [key for key in cmd_args if key.startswith("profiler.")]
        if profiler_keys:
            config_data["profiler"] = {}
            for key in profiler_keys:
                short_key = key.split(".")[-1]
                config_data["profiler"][short_key] = cmd_args[key]

    def _add_logging_config(self, config_data: Dict[str, Any], cmd_args: Dict[str, Any]) -> None:
        logging_keys = [key for key in cmd_args if key.startswith("logging.")]
        if logging_keys:
            config_data["logging"] = {}
            for key in logging_keys:
                short_key = key.split(".")[-1]
                config_data["logging"][short_key] = cmd_args[key]

    def _write_toml_config(self, cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun) -> str:
        config_path = tr.output_path / "config.toml"
        config_data = self._filter_config_data(cmd_args)
        with config_path.open("w") as toml_file:
            toml.dump(config_data, toml_file)
        return str(config_path)

    def generate_test_command(
        self, env_vars: Dict[str, str], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        self._write_toml_config(cmd_args, tr)
        return ["comm_replay", "--config /output/config.toml"]
