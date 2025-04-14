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

import json
import os
from typing import Dict, Optional, cast

from cloudai import ReportGenerationStrategy
from cloudai.systems.slurm import SlurmSystem

from .data.http_data_repository import HttpDataRepository
from .data.publisher import NeMoRunPublisher


class NeMoRunDataStoreReportGenerationStrategy(ReportGenerationStrategy):
    """Report generation strategy for NeMo 2.0 data store."""

    def can_handle_directory(self) -> bool:
        has_stdout = False
        has_report_data = False

        for _, __, files in os.walk(self.test_run.output_path):
            for file in files:
                if file == "stdout.txt":
                    has_stdout = True
                if file == "report_data.json":
                    has_report_data = True
                if has_stdout and has_report_data:
                    return True

        return False

    def generate_report(self) -> None:
        raw_data = self._load_data_file()
        if raw_data is None:
            return
        self._publish(raw_data)

    def _load_data_file(self) -> Optional[Dict]:
        data_file = self.test_run.output_path / "report_data.json"
        if not data_file.exists():
            return None
        with open(data_file, "r") as f:
            return json.load(f)

    def _publish(self, raw_data: Dict) -> None:
        slurm_system = cast(SlurmSystem, self.system)
        if slurm_system.data_repository is None:
            return

        repository = HttpDataRepository(
            slurm_system.data_repository.endpoint,
            slurm_system.data_repository.verify_certs,
        )
        publisher = NeMoRunPublisher(repository=repository)
        publisher.publish(raw_data)
