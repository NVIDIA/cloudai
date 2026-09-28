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

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class _SlurmStepMetadataBase(BaseModel):
    """Represents the metadata of a Slurm job step."""

    model_config = ConfigDict(extra="forbid")

    job_id: int
    name: str
    state: str
    start_time: str
    end_time: str
    elapsed_time_sec: int
    exit_code: str


class SlurmStepMetadata(_SlurmStepMetadataBase):
    """Represents the metadata of a Slurm job step."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    submit_line: str
    cluster_name: str = Field(default="", exclude=True)

    @classmethod
    def from_sacct_output(cls, output: str, delimiter: str) -> list[SlurmStepMetadata]:
        jobs: list[SlurmStepMetadata] = []
        for line in output.splitlines():
            if job := cls._from_sacct_single_line(line, delimiter):
                jobs.append(job)
        return jobs

    @classmethod
    def _from_sacct_single_line(cls, line: str, delimiter: str) -> SlurmStepMetadata | None:
        data = line.split(delimiter)
        if data and not data[-1]:
            data.pop()
        if len(data) < 8:
            return None

        job_id, step_id = data[0].split(".") if "." in data[0] else (data[0], "")
        has_cluster_name = len(data) >= 9
        cluster_name = data[7] if has_cluster_name else ""
        submit_line = delimiter.join(data[8:] if has_cluster_name else data[7:])

        return cls(
            job_id=int(job_id),
            step_id=step_id,
            name=data[1],
            state=data[2],
            exit_code=data[3],
            start_time=data[4],
            end_time=data[5],
            elapsed_time_sec=int(data[6]),
            cluster_name=cluster_name,
            submit_line=submit_line,
        )


class SlurmJobMetadata(_SlurmStepMetadataBase):
    """Represents the metadata of a Slurm job."""

    srun_cmd: str
    test_cmd: str
    is_single_sbatch: bool = False
    job_root: Path
    job_steps: list[SlurmStepMetadata]
    nodes: list[str] = Field(default_factory=list)

    @field_serializer("job_root")
    def _path_serializer(self, v: Path) -> str:
        return str(v)


class MetadataSystem(BaseModel):
    """Represents the system metadata."""

    os_type: str
    os_version: str
    linux_kernel_version: str
    gpu_arch_type: str
    gpu_count: int = 0
    gpu_inventory: str = "null"
    cpu_model_name: str
    cpu_arch_type: str
    cpu_vendor: str = "null"


class MetadataMPI(BaseModel):
    """Represents the MPI metadata."""

    mpi_type: str
    mpi_version: str
    hpcx_version: str


class MetadataCUDA(BaseModel):
    """Represents the CUDA metadata."""

    cuda_build_version: str
    cuda_runtime_version: str
    cuda_driver_version: str
    nvidia_driver_version: str = "null"
    cuda_toolkit_version: str = "null"


class MetadataNetwork(BaseModel):
    """Represents the network metadata."""

    nics: str
    nic_count: int = 0
    nic_inventory: str = "null"
    hca_firmware_versions: str = "null"
    switch_type: str
    network_name: str
    mofed_version: str
    doca_host_version: str = "null"
    libfabric_version: str


class MetadataNCCL(BaseModel):
    """Represents the NCCL metadata."""

    version: str
    commit_sha: str


class MetadataSlurm(BaseModel):
    """Represents the Slurm metadata."""

    cluster_name: str
    node_list: str
    num_nodes: str
    ntasks_per_node: str
    ntasks: str
    job_id: str


class SlurmSystemMetadata(BaseModel):
    """Represents the Slurm system metadata."""

    user: str
    system: MetadataSystem
    mpi: MetadataMPI
    cuda: MetadataCUDA
    network: MetadataNetwork
    nccl: MetadataNCCL
    slurm: MetadataSlurm
