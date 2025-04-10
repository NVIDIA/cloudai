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
from typing import Any, Dict, List, Optional, Union, cast

import toml
from pydantic import BaseModel, Field, root_validator

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
from cloudai.workloads.chakra_replay import ChakraReplayTestDefinition


def single(val: Union[str, List[str], Any]) -> Any:
    return val[0] if isinstance(val, list) else val


class RunConfig(BaseModel):
    """Run configuration."""

    warmup_iters: int
    iters: int


class TraceConfig(BaseModel):
    """Trace configuration."""

    directory: str


class TensorAllocatorConfig(BaseModel):
    """Tensor allocator configuration."""

    reuse_tensors: bool = True


class CommBackend(BaseModel):
    """Communication backend."""

    name: str = "pytorch-dist"
    backend: str = "nccl"


class CommConfig(BaseModel):
    """Communication configuration."""

    backend: CommBackend = Field(default_factory=CommBackend)


class ProfilerConfig(BaseModel):
    """Profiler configuration."""

    enabled: bool = False


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"


class ChakraReplayConfig(BaseModel):
    """ChakraReplay configuration."""

    run: RunConfig
    trace: Optional[TraceConfig]
    tensor_allocator: TensorAllocatorConfig
    comm: CommConfig
    profiler: ProfilerConfig
    logging: LoggingConfig

    @classmethod
    def _build_run(cls, values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "warmup_iters" not in values and "iters" not in values:
            return None
        run = {}
        if "warmup_iters" in values:
            run["warmup_iters"] = int(single(values["warmup_iters"]))
        if "iters" in values:
            run["iters"] = int(single(values["iters"]))
        return run

    @classmethod
    def _build_trace(cls, values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "trace_dir" not in values:
            return None
        return {"directory": single(values["trace_dir"])}

    @classmethod
    def _build_tensor_allocator(cls, values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "reuse_tensors" not in values:
            return None
        return {"reuse_tensors": str(single(values["reuse_tensors"])).lower() in {"1", "true", "yes", "on"}}

    @classmethod
    def _build_comm(cls, values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "backend.name" not in values and "backend.backend" not in values:
            return None
        backend = {}
        if "backend.name" in values:
            backend["name"] = single(values["backend.name"])
        if "backend.backend" in values:
            backend["backend"] = single(values["backend.backend"])
        return {"backend": backend}

    @classmethod
    def _build_profiler(cls, values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "profiler.enabled" not in values:
            return None
        return {"enabled": str(single(values["profiler.enabled"])).lower() in {"1", "true", "yes", "on"}}

    @classmethod
    def _build_logging(cls, values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "logging.level" not in values:
            return None
        return {"level": single(values["logging.level"])}

    @root_validator(pre=True)
    def build_nested(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        run = cls._build_run(values)
        if run is not None:
            values["run"] = run
        trace = cls._build_trace(values)
        if trace is not None:
            values["trace"] = trace
        tensor_allocator = cls._build_tensor_allocator(values)
        if tensor_allocator is not None:
            values["tensor_allocator"] = tensor_allocator
        comm = cls._build_comm(values)
        if comm is not None:
            values["comm"] = comm
        profiler = cls._build_profiler(values)
        if profiler is not None:
            values["profiler"] = profiler
        logging_ = cls._build_logging(values)
        if logging_ is not None:
            values["logging"] = logging_
        return values

    @classmethod
    def from_cmd_args(cls, cmd_args: Dict[str, Union[str, List[str]]]) -> "ChakraReplayConfig":
        return cls.model_validate(cmd_args)

    def write_to_toml(self, output_path: Path) -> Path:
        config_path = output_path / "config.toml"
        with config_path.open("w") as toml_file:
            toml.dump(self.dict(), toml_file)
        return config_path


class ChakraReplaySlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """ChakraReplay SLURM command generation strategy."""

    def _container_mounts(self, tr: TestRun) -> List[str]:
        tdef = cast(ChakraReplayTestDefinition, tr.test.test_definition)
        trace_dir = tdef.cmd_args.trace_dir

        if not trace_dir:
            return []

        replay_exec = tdef.comm_replay_executable
        if replay_exec.git_repo.installed_path is None:
            raise ValueError("installed_path should never be None")

        installed_path: Path = replay_exec.git_repo.installed_path.resolve()

        mounts = [
            f"{trace_dir}:{trace_dir}",
            f"{installed_path}:{installed_path}",
        ]

        return [",".join(mounts)]

    def _parse_slurm_args(
        self,
        job_name_prefix: str,
        env_vars: Dict[str, Union[str, List[str]]],
        cmd_args: Dict[str, Union[str, List[str]]],
        tr: TestRun,
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)
        tdef = cast(ChakraReplayTestDefinition, tr.test.test_definition)
        base_args.update({"image_path": str(tdef.docker_image.installed_path)})
        return base_args

    def _gen_srun_command(
        self,
        slurm_args: Dict[str, Any],
        env_vars: Dict[str, Union[str, List[str]]],
        cmd_args: Dict[str, Union[str, List[str]]],
        tr: TestRun,
    ) -> str:
        config = ChakraReplayConfig.from_cmd_args(cmd_args)
        config.write_to_toml(tr.output_path)
        tdef = cast(ChakraReplayTestDefinition, tr.test.test_definition)
        if tdef.comm_replay_executable.git_repo.installed_path is None:
            raise ValueError("installed_path should never be None")
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
        run_command = f'{" ".join(run_prefix)} bash -c "comm_replay --config /cloudai_run_results/config.toml"'

        return f"{install_command}\n{run_command}"
