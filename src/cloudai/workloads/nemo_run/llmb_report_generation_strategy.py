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

from typing import ClassVar, cast

from cloudai import ReportGenerationStrategy

from .data.http_data_repository import HttpDataRepository
from .data.llama_record_publisher import NeMoRunLLAMARecordPublisher


class NeMoRunLLMBReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from NeMoRun directories."""

    metrics: ClassVar[list[str]] = ["default", "step-time"]

    def generate_report(self) -> None:
        pass

    def publish_job_data(self) -> None:
        from cloudai.systems.slurm.slurm_system import SlurmSystem

        slurm_system = cast(SlurmSystem, self.system)
        if slurm_system.data_repository is None:
            return

        repository_instance = HttpDataRepository(
            slurm_system.data_repository.post_endpoint,
            slurm_system.data_repository.token,
            slurm_system.data_repository.index,
            slurm_system.data_repository.verify_certs,
        )
        publisher = NeMoRunLLAMARecordPublisher(repository=repository_instance)
        # TODO: Add data to publish
        publisher.publish({})
