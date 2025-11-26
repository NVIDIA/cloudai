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
from __future__ import annotations

import logging
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from cloudai.systems.slurm import SlurmCommandGenStrategy
from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import pandas as pd


class NIXLCmdGenBase(SlurmCommandGenStrategy):
    """Base command generation strategy for NIXL-based workloads."""

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
        cmd = [
            *self.gen_srun_prefix(),
            f"--output={self.test_run.output_path.absolute() / 'etcd.log'}",
            "--overlap",
            "--ntasks-per-node=1",
            "--ntasks=1",
            "--nodelist=$SLURM_JOB_MASTER_NODE",
            "-N1",
            *etcd_cmd,
            " &",
        ]
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
            f"kill -9 ${pid_var}\n",
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
        prefix_part = self.gen_srun_prefix()
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
