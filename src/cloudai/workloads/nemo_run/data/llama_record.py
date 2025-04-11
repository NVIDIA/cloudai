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

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SystemMetadata(BaseModel):
    """
    Hold system metadata common to all workloads.

    Attributes:
      cluster: The cluster identifier for the system.
      user: The user associated with the system.
      devices: The number of GPUs per node.
    """

    cluster: Optional[str] = Field(default=None, alias="s_cluster")
    user: Optional[str] = Field(default=None, alias="s_user")
    devices: Optional[int] = Field(default=None, alias="l_devices")

    model_config = ConfigDict(populate_by_name=True)


class JobMetadata(BaseModel):
    """
    Hold job metadata common to all workloads.

    Attributes:
      job_id: The job ID provided by the service.
      job_mode: The job mode (e.g. "training").
      image: The path or identifier for the software image.
      num_gpus: The number of GPUs in the job.
      num_nodes: The number of nodes used in the job.
    """

    job_id: Optional[str] = Field(default=None, alias="s_job_id")
    job_mode: Optional[str] = Field(default=None, alias="s_job_mode")
    image: Optional[str] = Field(default=None, alias="s_image")
    num_nodes: Optional[int] = Field(default=None, ge=1, alias="l_num_nodes")
    num_gpus: Optional[int] = Field(default=None, alias="l_num_gpus")

    model_config = ConfigDict(populate_by_name=True)


class GSWMetadata(BaseModel):
    """Hold GSW metadata."""

    gsw_version: str = Field(alias="s_gsw_version")

    model_config = ConfigDict(populate_by_name=True)


class NeMoRunContext(BaseModel):
    """Context for a NeMoRun workload."""

    timestamp: datetime = Field(alias="ts_timestamp")
    created_ts: int = Field(alias="ts_created")
    framework: str = Field(alias="s_framework")
    fw_version: str = Field(alias="s_fw_version")
    model: str = Field(alias="s_model")
    size: str = Field(alias="s_model_size")
    workload: str = Field(alias="s_workload")
    dtype: str = Field(alias="s_dtype")
    base_config: str = Field(alias="s_base_config")
    max_steps: int = Field(alias="l_max_steps")
    seq_len: int = Field(alias="l_seq_len")
    num_layers: int = Field(alias="l_num_layers")
    vocab_size: int = Field(alias="l_vocab_size")
    hidden_size: int = Field(alias="l_hidden_size")
    count: int = Field(alias="l_count")
    gbs: int = Field(alias="l_gbs")
    mbs: int = Field(alias="l_mbs")
    pp: int = Field(alias="l_pp")
    tp: int = Field(alias="l_tp")
    vp: int = Field(alias="l_vp")
    cp: int = Field(alias="l_cp")
    synthetic_dataset: bool = Field(alias="b_synthetic_dataset")

    model_config = ConfigDict(populate_by_name=True)


class NeMoRunPerformanceMetrics(BaseModel):
    """Performance metrics for a NemoRun workload."""

    metric: float = Field(alias="d_metric")
    metric_stddev: float = Field(alias="d_metric_stddev")
    step_time_mean: float = Field(alias="d_step_time_mean")
    tokens_per_sec: float = Field(alias="d_tokens_per_sec")
    checkpoint_size: Optional[int] = Field(default=None, alias="l_checkpoint_size")
    checkpoint_save_rank_time: Optional[float] = Field(alias="d_checkpoint_save_rank_time")

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class NeMoRunLLAMARecord(BaseModel):
    """LLAMA record for a NeMoRun workload."""

    system: SystemMetadata
    job: JobMetadata
    context: NeMoRunContext
    performance: NeMoRunPerformanceMetrics
    gsw: GSWMetadata

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def from_flat_dict(cls, data: dict) -> "NeMoRunLLAMARecord":
        performance_data = {k: v for k, v in data.items() if k.startswith("d_")}
        performance = NeMoRunPerformanceMetrics(**performance_data)
        exclude_context = {
            "s_job_id",
            "s_job_mode",
            "s_image",
            "l_devices",
            "l_num_nodes",
            "s_user",
            "s_cluster",
            "l_num_gpus",
            "s_gsw_version",
        }
        context_data = {
            k: v
            for k, v in data.items()
            if (k.startswith("ts_") or k.startswith("s_") or k.startswith("l_") or k == "b_synthetic_dataset")
            and k not in exclude_context
        }
        context = NeMoRunContext(**context_data)
        job_keys = {
            k: v
            for k, v in data.items()
            if k in {"s_job_id", "s_job_mode", "s_image", "l_num_gpus", "l_num_nodes", "s_cluster"}
        }
        job = JobMetadata(**job_keys)
        system_keys = {k: v for k, v in data.items() if k in {"s_user", "s_cluster", "l_devices"}}
        system = SystemMetadata(**system_keys)
        if "s_gsw_version" in data:
            gsw = GSWMetadata(**{"s_gsw_version": data["s_gsw_version"]})
        else:
            raise ValueError("Missing required key 's_gsw_version' for GSW metadata.")
        return cls(
            system=system,
            job=job,
            context=context,
            performance=performance,
            gsw=gsw,
        )

    def to_flat_dict(self) -> dict:
        flat = {}
        flat.update(self.context.model_dump(by_alias=True, exclude={"gsw"}))
        flat.update(self.performance.model_dump(by_alias=True))
        flat.update(self.job.model_dump(by_alias=True))
        flat.update(self.system.model_dump(by_alias=True))
        flat.update(self.gsw.model_dump(by_alias=True))
        return flat
