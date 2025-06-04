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

import shutil

from cloudai import ReportGenerationStrategy


class AIDynamoReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from AI Dynamo run directories."""

    def can_handle_directory(self) -> bool:
        output_path = self.test_run.output_path
        csv_path = output_path / "profile_genai_perf.csv"
        json_path = output_path / "profile_genai_perf.json"
        return csv_path.exists() and json_path.exists()

    def generate_report(self) -> None:
        output_path = self.test_run.output_path
        source_csv = output_path / "profile_genai_perf.csv"
        target_csv = output_path / "report.csv"

        shutil.copy2(source_csv, target_csv)

        # TODO: Add more fields to the report