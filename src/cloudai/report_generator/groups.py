# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dataclasses import dataclass

from cloudai.core import TestRun


@dataclass
class TRGroupItem:
    """Item in a group of TestRuns."""

    name: str
    tr: TestRun
    compact_name: str | None = None
    full_name: str | None = None

    @property
    def v2_compact_name(self) -> str:
        """Return the concise label used by v2 reports."""
        return self.compact_name or self.name

    @property
    def v2_full_name(self) -> str:
        """Return the detailed label used by v2 reports."""
        return self.full_name or self.v2_compact_name


@dataclass
class GroupedTestRuns:
    """Group of TestRuns."""

    name: str
    items: list[TRGroupItem]
