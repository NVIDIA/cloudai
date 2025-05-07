# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .._core.installables import GitRepo
from .._core.registry import Registry
from .workload import CmdArgs, NsysConfiguration


class TestSpecModel(BaseModel):
    """Model for test definition in test scenario."""

    __test__ = False

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    description: Optional[str] = None
    test_template_name: Optional[str] = None
    cmd_args: Optional[CmdArgs] = None
    extra_env_vars: dict[str, str] = {}
    extra_container_mounts: list[str] = []
    git_repos: list[GitRepo] = []
    nsys: Optional[NsysConfiguration] = None
    agent: str = "grid_search"
    agent_steps: int = 1
    agent_metric: str = "default"


class TestRunDependencyModel(BaseModel):
    """Model for test dependency in test scenario."""

    __test__ = False

    model_config = ConfigDict(extra="forbid")

    type: Literal["end_post_comp", "start_post_init", "start_post_comp"]
    id: str


class TestRunModel(BaseModel):
    """Model for test run in test scenario."""

    __test__ = False

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    test_name: Optional[str] = None
    num_nodes: Optional[int] = None
    nodes: list[str] = Field(default_factory=list)
    weight: int = 0
    iterations: int = 1
    sol: Optional[float] = None
    ideal_perf: float = 1.0
    time_limit: Optional[str] = None
    dependencies: list[TestRunDependencyModel] = Field(default_factory=list)
    test_spec: Optional[TestSpecModel] = None

    @model_validator(mode="after")
    def check_test_name_or_type_is_set(self):
        if self.test_name is None and self.test_spec is None:
            raise ValueError("Either 'test_name' or 'test_spec' must be set.")

        if not self.test_name:
            if self.test_spec and not self.test_spec.test_template_name:
                raise ValueError("'test_spec.test_template_name' must be set if 'test_name' is not set.")

            registry = Registry()
            if self.test_spec and self.test_spec.test_template_name not in registry.test_definitions_map:
                raise ValueError(
                    f"Test type '{self.test_spec.test_template_name}' not found in the test definitions. "
                    f"Possible values are: {', '.join(registry.test_definitions_map.keys())}"
                )
        else:
            if self.test_spec and self.test_spec.test_template_name:
                raise ValueError("'test_spec.test_template_name' must not be set if 'test_name' is set.")

        return self


class TestScenarioModel(BaseModel):
    """Model for test scenario."""

    __test__ = False

    model_config = ConfigDict(extra="forbid")

    name: str
    sol_path: Optional[str] = None
    job_status_check: bool = True
    tests: list[TestRunModel] = Field(alias="Tests", min_length=1)
    pre_test: Optional[str] = None
    post_test: Optional[str] = None

    @model_validator(mode="after")
    def check_no_self_dependency(self):
        """Check for circular dependencies in the test scenario."""
        for test_run in self.tests:
            for dep in test_run.dependencies:
                if dep.id == test_run.id:
                    raise ValueError(f"Test '{test_run.id}' must not depend on itself.")

        return self

    @model_validator(mode="after")
    def check_no_duplicate_ids(self):
        """Check for duplicate test ids in the test scenario."""
        test_ids = set()
        for tr in self.tests:
            if tr.id in test_ids:
                raise ValueError(f"Duplicate test id '{tr.id}' found in the test scenario.")
            test_ids.add(tr.id)

        return self

    @model_validator(mode="after")
    def check_all_dependencies_are_known(self):
        """Check that all dependencies are known."""
        test_ids = set(tr.id for tr in self.tests)
        for tr in self.tests:
            for dep in tr.dependencies:
                if dep.id not in test_ids:
                    raise ValueError(f"Dependency section '{dep.id}' not found for test '{tr.id}'.")

        return self
