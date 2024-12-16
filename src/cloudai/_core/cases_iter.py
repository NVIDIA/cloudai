# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

from .test_scenario import TestRun, TestScenario

if TYPE_CHECKING:
    from .base_runner import NewBaseRunner


class CasesIter(ABC, Iterator):
    @property
    @abstractmethod
    def has_more_cases(self) -> bool: ...

    @abstractmethod
    def __iter__(self) -> Iterator: ...

    @abstractmethod
    def __next__(self) -> TestRun: ...

    @abstractmethod
    def on_completed(self, tr: TestRun, runner: "NewBaseRunner") -> None: ...


class StaticCasesListIter(CasesIter):
    def __init__(self, test_scenario: TestScenario) -> None:
        self.test_scenario = test_scenario
        self.ready_for_run: list[TestRun] = []
        self.submitted: set[str] = set()
        self.completed: set[str] = set()
        self.stop_on_completion: dict[str, list[TestRun]] = {}

    @property
    def test_runs(self) -> list[TestRun]:
        return self.test_scenario.test_runs

    @property
    def has_more_cases(self) -> bool:
        return len(self.submitted) < len(self.test_runs)

    def __iter__(self) -> Iterator:
        not_submitted = [tr for tr in self.test_runs if tr.name not in self.submitted]
        for tr in not_submitted:
            if not tr.dependencies:
                self.ready_for_run.append(tr)
            elif tr.dependencies:
                if "start_post_comp" in tr.dependencies:
                    dep = tr.dependencies["start_post_comp"]
                    if dep.test_run.name in self.completed:
                        self.ready_for_run.append(tr)
                elif "end_post_comp" in tr.dependencies:
                    self.ready_for_run.append(tr)
                    dep = tr.dependencies["end_post_comp"]
                    self.stop_on_completion.setdefault(dep.test_run.name, []).append(tr)

        for tr in not_submitted:  # submit post_init dependencies right after main
            if tr.dependencies and "start_post_init" in tr.dependencies:
                dep = tr.dependencies["start_post_init"]
                if dep.test_run.name in {t.name for t in self.ready_for_run}:
                    self.ready_for_run.append(tr)

        return self

    def __next__(self) -> TestRun:
        if not self.ready_for_run:
            raise StopIteration

        self.submitted.add(self.ready_for_run[0].name)
        return self.ready_for_run.pop(0)

    def on_completed(self, tr: TestRun, runner: "NewBaseRunner") -> None:
        self.completed.add(tr.name)
        for tr_to_stop in self.stop_on_completion.get(tr.name, []):
            runner.kill_one(tr_to_stop)
