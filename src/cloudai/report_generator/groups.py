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

from dataclasses import dataclass

from cloudai.core import TestDefinition, TestRun

from .util import diff_test_runs


@dataclass
class TRGroupItem:
    """Item in a group of TestRuns."""

    name: str
    tr: TestRun


@dataclass
class GroupedTestRuns:
    """Group of TestRuns."""

    name: str
    items: list[TRGroupItem]


@dataclass
class TestRunsGrouper:
    """Group TestRuns based on cmd_args or/and extra_env_vars."""

    __test__ = False

    trs: list[TestRun]
    group_by: list[str]

    def get_value(self, tdef: TestDefinition, field: str) -> str:
        """Get field value for cmd_args or extra_env_vars."""
        if field.startswith("extra_env_vars."):
            f_name = field[len("extra_env_vars.") :]
            v = str(tdef.extra_env_vars.get(f_name))
        else:
            v = getattr(tdef.cmd_args, field)
        return v

    def group_name(self, trs: list[TestRun]) -> str:
        """
        Get group name for a list of TestRuns.

        Assume all test runs are grouped by the group_by fields, so take all the values from the first test run.
        """
        if not self.group_by:
            return "all-in-one"
        parts = [f"{field}={self.get_value(trs[0].test.test_definition, field)}" for field in self.group_by]
        return " ".join(parts).replace("extra_env_vars.", "")

    def create_group(self, trs: list[TestRun], group_idx: str = "0") -> GroupedTestRuns:
        diff = diff_test_runs(trs)
        items: list[TRGroupItem] = []
        for idx, _ in enumerate(trs):
            name = f"{group_idx}.{idx}"
            if diff:
                item_name_parts = [f"{field}={vals[idx]}" for field, vals in diff.items()]
                name = " ".join(item_name_parts).replace("extra_env_vars.", "")
            items.append(TRGroupItem(name=name, tr=trs[idx]))
        return GroupedTestRuns(name=self.group_name(trs), items=items)

    def groups(self) -> list[GroupedTestRuns]:
        if not self.group_by:
            return [self.create_group(self.trs)]

        groups: list[list[TestRun]] = []
        for tr in self.trs:
            for group in groups:
                matched = all(
                    self.get_value(tr.test.test_definition, field)
                    == self.get_value(group[0].test.test_definition, field)
                    for field in self.group_by
                )

                if matched:
                    group.append(tr)
                    break
            else:  # runs only if no break happened
                groups.append([tr])

        res: list[GroupedTestRuns] = []
        for grp_idx, group in enumerate(groups):
            res.append(self.create_group(group, group_idx=str(grp_idx)))
        return res
