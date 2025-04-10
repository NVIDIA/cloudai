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


class SystemContext(BaseModel):
    """
    Hold system metadata common to all workloads.

    Attributes:
      cluster: The cluster identifier for the system.
      user: The user associated with the system.
      num_gpus: The number of GPUs associated with the system.
    """

    cluster: Optional[str] = Field(default=None, alias="s_cluster")
    user: Optional[str] = Field(default=None, alias="s_user")
    num_gpus: Optional[int] = Field(default=None, alias="l_num_gpus")

    model_config = {
        "populate_by_name": True,
    }


class JobContext(BaseModel):
    """
    Hold job metadata common to all workloads.

    Attributes:
      job_id: The job id provided by the service provider.
      image: The path or identifier for the software image.
      num_nodes: Number of nodes used in the job.
    """

    job_id: Optional[str] = Field(default=None, alias="s_job_id")
    image: Optional[str] = Field(default=None, alias="s_image")
    num_nodes: Optional[int] = Field(default=None, ge=1, alias="l_num_nodes")

    model_config = {
        "populate_by_name": True,
    }


class GSWContext(BaseModel):
    """
    Hold GSW metadata common to all workloads.

    Attributes:
      gsw_version: The version of the GSW.
    """

    gsw_version: str = Field(alias="s_gsw_version")

    model_config = {
        "populate_by_name": True,
    }


class NeMoRunContext(BaseModel):
    """NeMoRun context for a NemoRun workload."""

    timestamp: datetime = Field(alias="ts_timestamp")
    created_ts: int = Field(alias="ts_created")
    gsw: GSWContext

    framework: str = Field(alias="s_framework")
    fw_version: str = Field(alias="s_fw_version")
    model: str = Field(alias="s_model")
    model_size: str = Field(alias="s_model_size", pattern=r"[\d]+[mbtMBT]")
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

    model_config = {
        "populate_by_name": True,
    }


class NeMoRunPerformanceMetrics(BaseModel):
    """Performance metrics for NemoRun workloads."""

    metric: float = Field(alias="d_metric")
    metric_stddev: float = Field(alias="d_metric_stddev")
    step_time_mean: float = Field(alias="d_step_time_mean")
    tokens_per_sec: float = Field(alias="d_tokens_per_sec")
    checkpoint_size: Optional[int] = Field(default=None, alias="l_checkpoint_size")
    checkpoint_save_rank_time: Optional[float] = Field(alias="d_checkpoint_save_rank_time")

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class NeMoRunLLAMARecord(BaseModel):
    """NeMoRun LLAMA record for a NemoRun workload."""

    system: SystemContext
    job: JobContext
    context: NeMoRunContext
    performance: NeMoRunPerformanceMetrics

    @classmethod
    def from_flat_dict(cls, data: dict) -> "NeMoRunLLAMARecord":
        perf = {k: v for k, v in data.items() if k.startswith("d_")}
        performance = NeMoRunPerformanceMetrics(**perf)

        context_keys = {}
        for key in [
            "s_base_config",
            "s_framework",
            "s_fw_version",
            "s_model",
            "s_model_size",
            "s_workload",
            "s_dtype",
        ]:
            if key in data:
                context_keys[key] = data[key]
        for key in data:
            if key.startswith("l_") and key not in {"l_devices", "l_num_nodes"}:
                context_keys[key] = data[key]
        context = NeMoRunContext(**context_keys)

        job_keys = {}
        for key in ["s_job_id", "s_job_mode", "s_cluster", "s_image", "l_devices", "l_num_nodes"]:
            if key in data:
                job_keys[key] = data[key]
        job = JobContext(**job_keys)

        system_keys = {}
        for key in ["s_user", "s_cluster"]:
            if key in data:
                system_keys[key] = data[key]
        system = SystemContext(**system_keys)

        record = cls(context=context, job=job, system=system, performance=performance)
        return record

    def to_flat_dict(self) -> dict:
        flat = self.context.model_dump(by_alias=True, exclude={"performance", "job", "system", "gsw"})
        flat.update(self.performance.model_dump(by_alias=True))
        flat.update(self.job.model_dump(by_alias=True))
        flat.update(self.system.model_dump(by_alias=True))
        return flat
