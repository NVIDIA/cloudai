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

from pydantic import BaseModel, ConfigDict


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

    @classmethod
    def from_sacct_single_line(cls, line: str, delimiter: str) -> SlurmStepMetadata:
        data = line.split(delimiter)
        if len(data) < 8:
            raise ValueError(f"Invalid line: {line}")

        job_id, step_id = data[0].split(".") if "." in data[0] else (data[0], "")

        return cls(
            job_id=int(job_id),
            step_id=step_id,
            name=data[1],
            state=data[2],
            exit_code=data[3],
            start_time=data[4],
            end_time=data[5],
            elapsed_time_sec=int(data[6]),
            submit_line=data[7],
        )


class SlurmJobMetadata(_SlurmStepMetadataBase):
    """Represents the metadata of a Slurm job."""

    srun_cmd: str
    test_cmd: str
    is_single_sbatch: bool = False
    job_steps: list[SlurmStepMetadata]


class MetadataSystem(BaseModel):
    """Represents the system metadata."""

    os_type: str
    os_version: str
    linux_kernel_version: str
    gpu_arch_type: str
    cpu_model_name: str
    cpu_arch_type: str


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


class MetadataNetwork(BaseModel):
    """Represents the network metadata."""

    nics: str
    switch_type: str
    network_name: str
    mofed_version: str
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
