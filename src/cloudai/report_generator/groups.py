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

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from cloudai.core import TestRun

from .util import diff_test_runs

T = TypeVar("T")


@dataclass
class GroupItem(Generic[T]):
    """Generic item in a group."""

    name: str
    item: T


@dataclass
class GroupedItems(Generic[T]):
    """Generic group of items."""

    name: str
    items: list[GroupItem[T]]


@dataclass
class ItemsGrouper(Generic[T]):
    """Generic grouper for TestRuns or items containing TestRuns."""

    __test__ = False

    items: list[T]
    group_by: list[str]

    def _tr(self, item: T) -> TestRun:
        """Extract TestRun from item."""
        if isinstance(item, TestRun):
            return item
        if hasattr(item, "test_run"):
            return cast(TestRun, getattr(item, "test_run"))  # noqa: B009

        raise ValueError(f"Item {item} does not have a test_run attribute")

    def get_value(self, item: T, field: str) -> str:
        """Get field value for cmd_args or extra_env_vars."""
        tr = self._tr(item)
        tdef = tr.test.test_definition
        if field.startswith("extra_env_vars."):
            f_name = field[len("extra_env_vars.") :]
            v = str(tdef.extra_env_vars.get(f_name))
        else:
            v = str(getattr(tdef.cmd_args, field, ""))
        return v

    def group_name(self, items: list[T]) -> str:
        """
        Get group name for a list of items.

        Assume all items are grouped by the group_by fields, so take values from the first item.
        """
        if not self.group_by:
            return "all-in-one"
        parts = [f"{field}={self.get_value(items[0], field)}" for field in self.group_by]
        return " ".join(parts).replace("extra_env_vars.", "")

    def create_group(self, items: list[T], group_idx: str = "0") -> GroupedItems[T]:
        """Create a group from items."""
        trs = [self._tr(item) for item in items]
        diff = diff_test_runs(trs)

        group_items: list[GroupItem[T]] = []
        for idx, item in enumerate(items):
            name = f"{group_idx}.{idx}"
            if diff:
                item_name_parts = [f"{field}={vals[idx]}" for field, vals in diff.items()]
                name = " ".join(item_name_parts).replace("extra_env_vars.", "")
            group_items.append(GroupItem(name=name, item=item))

        return GroupedItems(name=self.group_name(items), items=group_items)

    def groups(self) -> list[GroupedItems[T]]:
        """Group items based on group_by fields."""
        if not self.group_by:
            return [self.create_group(self.items)]

        groups_list: list[list[T]] = []
        for item in self.items:
            for group in groups_list:
                matched = all(self.get_value(item, field) == self.get_value(group[0], field) for field in self.group_by)

                if matched:
                    group.append(item)
                    break
            else:  # runs only if no break happened
                groups_list.append([item])

        result: list[GroupedItems[T]] = []
        for grp_idx, group in enumerate(groups_list):
            result.append(self.create_group(group, group_idx=str(grp_idx)))
        return result


GroupedTestRuns = GroupedItems[TestRun]
TestRunsGrouper = ItemsGrouper[TestRun]
