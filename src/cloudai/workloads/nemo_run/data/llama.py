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


from pydantic import BaseModel, ConfigDict, Field

from cloudai.data.model.gsw_context import GSWContext
from cloudai.data.model.job_context import JobContext
from cloudai.data.model.system_context import SystemContext

from .base import NemoRunBaseRecord, PerformanceMetrics


class NeMoRunLLAMAContext(BaseModel):
    """NeMoRun LLAMA context for a NemoRun workload."""

    model_config = ConfigDict(protected_namespaces=())

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


class NeMoRunLLAMARecord(NemoRunBaseRecord):
    """NeMoRun LLAMA record for a NemoRun workload."""

    context: NeMoRunLLAMAContext
    job: JobContext
    system: SystemContext

    @classmethod
    def from_flat_dict(cls, data: dict) -> "NeMoRunLLAMARecord":
        top_level = {
            "ts_timestamp": data["ts_timestamp"],
            "b_synthetic_dataset": data["b_synthetic_dataset"],
            "ts_created": data["ts_created"],
        }
        perf = {k: v for k, v in data.items() if k.startswith("d_")}
        performance = PerformanceMetrics(**perf)

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
        context = NeMoRunLLAMAContext(**context_keys)

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

        gsw = GSWContext(s_gsw_version=data["s_gsw_version"])

        record = cls(**top_level, gsw=gsw, performance=performance, context=context, job=job, system=system)
        return record

    def to_flat_dict(self) -> dict:
        flat = self.model_dump(by_alias=True, exclude={"performance", "context", "job", "system", "gsw"})
        flat.update(self.gsw.model_dump(by_alias=True))
        flat.update(self.performance.model_dump(by_alias=True))
        flat.update(self.context.model_dump(by_alias=True))
        flat.update(self.job.model_dump(by_alias=True))
        flat.update(self.system.model_dump(by_alias=True))
        return flat
