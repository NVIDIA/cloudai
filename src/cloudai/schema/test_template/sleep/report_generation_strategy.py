#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

from cloudai import ReportGenerationStrategy


class SleepReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from sleep directories."""

    def can_handle_directory(self, directory_path: str) -> bool:
        return False

    def generate_report(self, test_name: str, directory_path: str, sol: Optional[float] = None) -> None:
        pass
