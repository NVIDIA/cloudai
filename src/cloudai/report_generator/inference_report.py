# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import abc
import dataclasses
import typing

import cloudai.metrics

if typing.TYPE_CHECKING:
    import cloudai.core


@dataclasses.dataclass(kw_only=True)
class InferenceRoleConfig:
    """Configured serving layout for one role, such as prefill or decode."""

    model: str | None = None
    num_nodes: int | None = None
    num_instances: int | None = None
    nodes_per_instance: int | None = None
    instances_per_node: int | None = None
    gpu_ids: list[int] | None = None
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    data_parallel_size: int | None = None
    expert_parallel: bool | None = None
    precision: str | None = None
    quantization: str | None = None


@dataclasses.dataclass(kw_only=True)
class InferenceConfig:
    """Resolved input settings for one measurement; unknown settings remain None."""

    model: str
    num_nodes: int
    backend: str | None = None
    benchmark: str | None = None
    phase: str | None = None
    container_image: str | None = None
    gpus_per_node: int | None = None
    serving_gpu_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    concurrency: int | None = None
    users: int | None = None
    request_count: int | None = None
    request_rate: float | None = None
    prefill: InferenceRoleConfig | None = None
    decode: InferenceRoleConfig | None = None
    shared_nodes: bool | None = None

    def context(self) -> dict[str, typing.Any]:
        """Display resolved input settings without inspecting workload-specific configurations."""
        labels = {
            "model": "Model",
            "num_nodes": "Nodes",
            "backend": "Backend",
            "container_image": "Container image",
            "serving_gpu_count": "Serving GPUs",
            "input_tokens": "Input tokens",
            "output_tokens": "Output tokens",
            "concurrency": "Concurrency",
            "users": "Users",
            "request_count": "Requests",
            "request_rate": "Request rate",
            "shared_nodes": "Prefill and decode share nodes",
        }
        context = {label: getattr(self, key) for key, label in labels.items() if getattr(self, key) is not None}
        if self.gpus_per_node is not None and self.gpus_per_node > 0:
            context["Allocated GPUs"] = self.num_nodes * self.gpus_per_node
        role_labels = {
            "model": "Model",
            "num_nodes": "Nodes",
            "num_instances": "Instances",
            "nodes_per_instance": "Nodes per instance",
            "instances_per_node": "Instances per node",
            "gpu_ids": "GPU IDs",
            "tensor_parallel_size": "TP",
            "pipeline_parallel_size": "PP",
            "data_parallel_size": "DP",
            "expert_parallel": "Expert parallel",
            "precision": "Precision",
            "quantization": "Quantization",
        }
        decode_label = "Decode" if self.prefill is not None else "Serving"
        for name, role in (("Prefill", self.prefill), (decode_label, self.decode)):
            if role is not None:
                context.update(
                    {
                        f"{name} / {label}": getattr(role, key)
                        for key, label in role_labels.items()
                        if getattr(role, key) is not None
                    }
                )
        return context


@dataclasses.dataclass(kw_only=True)
class InferenceReport:
    """One independently measured configuration, including all metric dimensions."""

    config: InferenceConfig
    observations: list[cloudai.metrics.MetricObservation] = dataclasses.field(default_factory=list)


class InferenceReportProvider(abc.ABC):
    """Interface implemented by inference test definitions."""

    @abc.abstractmethod
    def inference_reports(self, system: cloudai.core.System, tr: cloudai.core.TestRun) -> list[InferenceReport]:
        """Return one report per measurement, or an empty list when no results are available."""
