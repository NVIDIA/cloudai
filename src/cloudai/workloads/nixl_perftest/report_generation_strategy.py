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

from typing import ClassVar

from cloudai.core import ReportGenerationStrategy


class NIXLKVBenchDummyReport(ReportGenerationStrategy):
    """Dummy report to support sweeps as it requires "default" metric."""

    metrics: ClassVar[list[str]] = ["default"]

    def can_handle_directory(self) -> bool:
        return True

    def generate_report(self) -> None:
        pass
