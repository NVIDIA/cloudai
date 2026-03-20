# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from __future__ import annotations

import logging
import re
import shutil
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Generic, TypeVar, cast

from pydantic import BaseModel, Field, field_validator

from cloudai.core import DockerImage, Installable, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition
from cloudai.systems.slurm import SlurmCommandGenStrategy
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import pandas as pd


BUFFER_SIZE_FORMAT: Final[re.Pattern[str]] = re.compile(r"^(?P<num>\d+)(?P<unit>(b|kb|mb|gb)?)$")
DEVICE_FORMAT: Final[re.Pattern[str]] = re.compile(r"^\d+:[A-Z]:/[/\da-zA-Z._-]+$")
# 8gb is the default value in the nixl itself
# it's not set as a default in the model below to not propagate it into the srun if the user didn't explicitly set it
DEFAULT_TOTAL_BUFFER_SIZE = 8 * 1024 * 1024 * 1024


class NIXLBaseCmdArgs(CmdArgs):
    """Command line arguments for a NIXL workloads."""

    docker_image_url: str = Field(description="URL of the Docker image to use for the benchmark.")
    etcd_path: str = Field(default="etcd", description="Path to the etcd executable.")
    wait_etcd_for: int = Field(default=60, description="Number of seconds to wait for etcd to become healthy.")
    etcd_image_url: str | None = Field(
        default=None,
        description=(
            "Optional URL of the Docker image to use for etcd, by default etcd will be run from the same image "
            "as the benchmark."
        ),
    )


class NIXLExtendedCmdArgs(BaseModel):
    """Extended CLI for NIXL workloads. Used by nixl-bench and nixl-kvbench but not by nixl-perftest."""

    filepath: str | None = Field(
        default=None,
        description="Directory path (in container) for storage operations. Example: /data",
    )
    total_buffer_size: str | list[str] | None = Field(
        default=None,
        description=(
            "Total buffer size in bytes. Examples: 1024, 1kb, 1mb, 1gb. Use with device_list. The size will be passed "
            "into NIXL as integer (bytes)"
        ),
    )
    device_list: str | list[str] | None = Field(
        default=None,
        description="Device specs in format 'id:type:path' (e.g., '11:F:/store0.bin,27:K:/dev/nvme0n1')",
    )

    @field_validator("filepath", mode="after")
    @classmethod
    def validate_filepath(cls, v: str | None) -> str | None:
        if v is None:
            return None

        if not Path(v).is_absolute():
            logging.warning(
                f"Provided container path {v!s} is not absolute. Prepending '/' to make it absolute within container."
            )
            return "/" + v

        return v

    @field_validator("total_buffer_size", mode="before")
    @classmethod
    def prevalidate_total_buffer_size(cls, v: Any) -> Any:
        """Handle integers."""
        if v is None:
            return v
        elif isinstance(v, list):
            return list(map(str, v))
        else:
            return str(v)

    @field_validator("total_buffer_size", mode="after")
    @classmethod
    def validate_total_buffer_size(cls, v: str | list[str] | None) -> str | list[str] | None:
        if not v:
            return None
        elif isinstance(v, list):
            return list(map(parse_total_buffer_size, v))
        else:
            return parse_total_buffer_size(v)

    @field_validator("device_list", mode="after")
    @classmethod
    def validate_device_list(cls, v: str | list[str] | None) -> str | list[str] | None:
        if not v:
            return None
        elif isinstance(v, list):
            return list(map(parse_device, v))
        else:
            return parse_device(v)


NIXLCmdArgsT = TypeVar("NIXLCmdArgsT", bound=NIXLBaseCmdArgs)


class NIXLBaseTestDefinition(TestDefinition, Generic[NIXLCmdArgsT]):
    """Test definition for a NIXL workloads."""

    cmd_args: NIXLCmdArgsT
    _nixl_image: DockerImage | None = None
    _etcd_image: DockerImage | None = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._nixl_image:
            self._nixl_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._nixl_image

    @property
    def etcd_image(self) -> DockerImage | None:
        if not self.cmd_args.etcd_image_url:
            return None
        if not self._etcd_image:
            self._etcd_image = DockerImage(url=self.cmd_args.etcd_image_url)
        return self._etcd_image

    @property
    def installables(self) -> list[Installable]:
        installables = [self.docker_image, *self.git_repos]
        if self.etcd_image:
            installables.append(self.etcd_image)
        return installables


class NIXLCmdGenBase(SlurmCommandGenStrategy):
    """Base command generation strategy for NIXL-based workloads."""

    def __init__(self, system: SlurmSystem, test_run: TestRun) -> None:
        super().__init__(system, test_run)
        self._current_image_url: str | None = None

    def image_path(self) -> str | None:
        return self._current_image_url

    def _container_mounts(self) -> list[str]:
        mounts = []
        mounts.extend(self._filepath_mounts())
        mounts.extend(self._device_list_mounts())
        return mounts

    def _filepath_mounts(self) -> list[str]:
        filepath_raw: str | None = cast(str | None, self.test_run.test.cmd_args_dict.get("filepath"))
        if not filepath_raw:
            return []

        filepath = Path(filepath_raw)

        local_dir = self.test_run.output_path / "filepath_mount" / filepath.name
        if local_dir.exists() and not local_dir.is_dir():
            raise ValueError(f"Expected a directory for filepath mount, but found file at {local_dir}.")
        local_dir.mkdir(parents=True, exist_ok=True)
        return [f"{local_dir.absolute()}:{filepath}"]

    def _device_list_mounts(self) -> list[str]:
        device_list_raw: str | None = cast(str | None, self.test_run.test.cmd_args_dict.get("device_list"))
        if not device_list_raw:
            return []

        file_devices = get_files_from_device_list(device_list_raw)
        if not file_devices:
            return []

        if "total_buffer_size" in self.test_run.test.cmd_args_dict:
            total_buffer_size = int(cast(str, self.test_run.test.cmd_args_dict["total_buffer_size"]))
        else:
            total_buffer_size = DEFAULT_TOTAL_BUFFER_SIZE

        mounts = []
        used_filenames: set[str] = set()
        for device_path in file_devices:
            unique_device_filename = self._unique_file_name(device_path.name, used_filenames)
            local_device_path = self.test_run.output_path / "device_list_mounts" / unique_device_filename
            self._ensure_device_file(local_device_path, total_buffer_size)
            mounts.append(f"{local_device_path.absolute()}:{device_path}")
        return mounts

    def _ensure_device_file(self, file_path: Path, size: int) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists() and file_path.is_dir():
            raise ValueError(f"Expected a file for device_list: {file_path.name}, but found directory at {file_path}.")

        if file_path.exists() and file_path.stat().st_size == size:
            return

        with file_path.open("wb") as f:
            f.truncate(size)

    def _unique_file_name(self, file_name: str, used_filenames: set[str]) -> str:
        if file_name not in used_filenames:
            used_filenames.add(file_name)
            return file_name

        base = Path(file_name).stem
        suffix = Path(file_name).suffix
        idx = 1
        candidate = f"{base}_{idx}{suffix}"
        while candidate in used_filenames:
            idx += 1
            candidate = f"{base}_{idx}{suffix}"

        used_filenames.add(candidate)
        return candidate

    def cleanup_job_artifacts(self) -> None:
        for cleanup_target in self._cleanup_targets():
            if cleanup_target.exists():
                shutil.rmtree(cleanup_target)
                logging.debug(f"Cleaned up job artifact: {cleanup_target}")

    def _cleanup_targets(self) -> list[Path]:
        cleanup_targets: list[Path] = []

        filepath_raw: str | None = cast(str | None, self.test_run.test.cmd_args_dict.get("filepath"))
        if filepath_raw:
            cleanup_targets.append((self.test_run.output_path / "filepath_mount").resolve())

        device_list_raw: str | None = cast(str | None, self.test_run.test.cmd_args_dict.get("device_list"))
        if device_list_raw and get_files_from_device_list(device_list_raw):
            cleanup_targets.append((self.test_run.output_path / "device_list_mounts").resolve())

        return cleanup_targets

    @property
    def final_env_vars(self) -> dict[str, str | list[str]]:
        env_vars = super().final_env_vars
        env_vars["NIXL_ETCD_NAMESPACE"] = "/nixl/kvbench/$(uuidgen)"
        env_vars["NIXL_ETCD_ENDPOINTS"] = '"$SLURM_JOB_MASTER_NODE:2379"'
        env_vars["SLURM_JOB_MASTER_NODE"] = "$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)"
        return env_vars

    @final_env_vars.setter
    def final_env_vars(self, value: dict[str, str | list[str]]) -> None:
        super().final_env_vars = value

    def gen_etcd_srun_command(self, etcd_path: str) -> list[str]:
        etcd_cmd = [
            etcd_path,
            "--listen-client-urls=http://0.0.0.0:2379",
            "--advertise-client-urls=http://$SLURM_JOB_MASTER_NODE:2379",
            "--listen-peer-urls=http://0.0.0.0:2380",
            "--initial-advertise-peer-urls=http://$SLURM_JOB_MASTER_NODE:2380",
            '--initial-cluster="default=http://$SLURM_JOB_MASTER_NODE:2380"',
            "--initial-cluster-state=new",
        ]
        tdef = cast(NIXLBaseTestDefinition[NIXLBaseCmdArgs], self.test_run.test)
        curr_image = self._current_image_url
        if tdef.etcd_image:
            self._current_image_url = str(tdef.etcd_image.installed_path)
        cmd = [
            *self.gen_srun_prefix(with_num_nodes=False),
            f"--output={self.test_run.output_path.absolute() / 'etcd.log'}",
            "--overlap",
            "--ntasks-per-node=1",
            "--ntasks=1",
            "--nodelist=$SLURM_JOB_MASTER_NODE",
            "-N1",
            *etcd_cmd,
            " &",
        ]
        self._current_image_url = curr_image
        return cmd

    def gen_wait_for_etcd_command(self, timeout: int = 60) -> list[str]:
        cmd = [
            "timeout",
            str(timeout),
            "bash",
            "-c",
            '"until curl -s $NIXL_ETCD_ENDPOINTS/health > /dev/null 2>&1; do sleep 1; done" || {\n',
            f'  echo "ETCD ($NIXL_ETCD_ENDPOINTS) was unreachable after {timeout} seconds";\n',
            "  exit 1\n",
            "}",
        ]
        return cmd

    def gen_kill_and_wait_cmd(self, pid_var: str, timeout: int = 60) -> list[str]:
        cmd = [
            f"kill -TERM ${pid_var}\n",
            "timeout",
            str(timeout),
            "bash",
            "-c",
            f'"while kill -0 ${pid_var} 2>/dev/null; do sleep 1; done" || {{\n',
            f'  echo "Failed to kill ETCD (pid=${pid_var}) within {timeout} seconds";\n',
            "  exit 1\n",
            "}",
        ]
        return cmd

    def gen_nixlbench_srun_commands(self, test_cmd: list[str], backend: str) -> list[list[str]]:
        prefix_part = self.gen_srun_prefix(with_num_nodes=False)
        bash_part = [
            "bash",
            "-c",
            f'"source {(self.test_run.output_path / "env_vars.sh").absolute()}; {" ".join(test_cmd)}"',
        ]
        tpn_part = ["--ntasks-per-node=1", "--ntasks=1", "-N1"]

        cmds = [
            [*prefix_part, "--overlap", "--nodelist=$SLURM_JOB_MASTER_NODE", *tpn_part, *bash_part],
        ]

        if backend.upper() == "UCX":
            nnodes, _ = self.get_cached_nodes_spec()
            if nnodes > 1:
                cmds = [
                    [*prefix_part, "--overlap", f"--relative={idx}", *tpn_part, *bash_part] for idx in range(nnodes)
                ]
            else:
                cmds *= max(2, nnodes)

        return cmds

    def create_env_vars_file(self) -> None:
        with (self.test_run.output_path / "env_vars.sh").open("w") as f:
            for key, value in self.final_env_vars.items():
                if key in {"NIXL_ETCD_ENDPOINTS", "NIXL_ETCD_NAMESPACE"}:
                    continue
                if key == "SLURM_JOB_MASTER_NODE":  # this is an sbatch-level variable, not needed per-node
                    continue
                f.write(f"export {key}={value}\n")


@cache
def extract_nixlbench_data(stdout_file: Path) -> pd.DataFrame:
    if not stdout_file.exists():
        logging.debug(f"{stdout_file} not found")
        return lazy.pd.DataFrame()

    header_present, data = False, []
    for line in stdout_file.read_text().splitlines():
        if not header_present and (
            "Block Size (B)      Batch Size     " in line and "Avg Lat. (us)" in line and "B/W (GB/Sec)" in line
        ):
            header_present = True
            continue
        parts = line.split()
        if header_present and (len(parts) == 6 or len(parts) == 10):
            try:
                int(parts[0])  # block size
                int(parts[1])  # batch size
            except ValueError:
                # doesn't look like a data line, skip
                continue

            if len(parts) == 6:
                data.append([parts[0], parts[1], parts[2], parts[-1]])
            else:
                data.append([parts[0], parts[1], parts[3], parts[2]])

    df = lazy.pd.DataFrame(data, columns=["block_size", "batch_size", "avg_lat", "bw_gb_sec"])
    df["block_size"] = df["block_size"].astype(int)
    df["batch_size"] = df["batch_size"].astype(int)
    df["avg_lat"] = df["avg_lat"].astype(float)
    df["bw_gb_sec"] = df["bw_gb_sec"].astype(float)

    return df


def parse_device(v: str) -> str:
    for device in v.split(","):
        if not re.fullmatch(DEVICE_FORMAT, device):
            raise ValueError(f"Invalid device spec: {device}, must be in format 'id:type:path'")

    return v


def parse_total_buffer_size(v: str) -> str:
    multipliers: dict[str, int] = {"": 1, "b": 1, "kb": 2**10, "mb": 2**20, "gb": 2**30}

    match = re.fullmatch(BUFFER_SIZE_FORMAT, v)
    if match is None:
        raise ValueError(f"Could not parse total_buffer_size={v!r}.")

    amount = int(match.group("num"))
    unit = match.group("unit").lower()

    return str(amount * multipliers[unit])


def get_files_from_device_list(device_list: str) -> list[Path]:
    """
    Filter device_list into files and return their container paths.

    Expects validated device_list here
    """
    parsed: list[Path] = []

    for device_str in device_list.split(","):
        device_parts = device_str.split(":", 2)

        _, device_type, device_path = device_parts
        if device_type != "F":
            continue

        parsed.append(Path(device_path))

    return parsed
