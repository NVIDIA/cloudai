# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
import itertools
import pathlib
from typing import TYPE_CHECKING, Any, Optional, Type, TypeAlias

import cloudai.metrics

from ..models.workload import TestDefinition
from ..util import flatten_dict
from .registry import Registry
from .system import System

if TYPE_CHECKING:
    from ..models.scenario import TestScenario
    from .report_generation_strategy import ReportGenerationStrategy


class MetricErrorSentinel:
    """Singleton returned by report strategies on failure; use ``v is METRIC_ERROR`` to detect errors."""

    __slots__ = ()
    _instance: MetricErrorSentinel | None = None

    def __new__(cls) -> MetricErrorSentinel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "METRIC_ERROR"

    def __float__(self) -> float:
        return -1.0


METRIC_ERROR = MetricErrorSentinel()

MetricValue: TypeAlias = float | MetricErrorSentinel


class TestDependency:
    """
    Represents a dependency for a test.

    Attributes
        test_run (TestRun): TestRun object it depends on.
    """

    __test__ = False

    def __init__(self, test_run: "TestRun") -> None:
        """
        Initialize a TestDependency instance.

        Args:
            test_run (TestRun): TestRun object it depends on.
        """
        self.test_run = test_run


class TestRun:
    """Mutable execution state bound to one prepared test definition."""

    __test__ = False

    def __init__(
        self,
        test: TestDefinition,
        nodes: list[str] | None = None,
        output_path: pathlib.Path = pathlib.Path(""),
        current_iteration: int = 0,
        step: int = 0,
        dependencies: dict[str, TestDependency] | None = None,
        pre_test: TestScenario | None = None,
        post_test: TestScenario | None = None,
    ) -> None:
        self.test = test
        self.nodes = list(nodes if nodes is not None else test.nodes)
        self.output_path = output_path
        self.current_iteration = current_iteration
        self.step = step
        self.dependencies = dependencies if dependencies is not None else {}
        self.pre_test = pre_test
        self.post_test = post_test

    @property
    def name(self) -> str:
        return self.test.id or self.test.name

    @property
    def num_nodes(self) -> int | list[int]:
        return self.test.num_nodes if self.test.num_nodes is not None else 1

    @property
    def num_nodes_explicit(self) -> bool:
        return self.test.num_nodes is not None and "num_nodes" in self.test.model_fields_set

    @property
    def time_limit(self) -> str | None:
        return self.test.time_limit

    @property
    def sol(self) -> float | None:
        return self.test.sol if isinstance(self.test.sol, (float, int)) else None

    @property
    def metric_sol(self) -> cloudai.metrics.MetricSOLConfig:
        return self.test.metric_sol

    @property
    def weight(self) -> float:
        return self.test.weight

    @property
    def reports(self) -> set[type[ReportGenerationStrategy]]:
        return self.test.reporters

    @property
    def exclude_nodes(self) -> list[str]:
        return self.test.exclude_nodes

    @property
    def iterations(self) -> int:
        return self.test.iterations

    @property
    def ideal_perf(self) -> float:
        return self.test.ideal_perf

    @property
    def extra_srun_args(self) -> str | None:
        return self.test.srun_args

    @property
    def pin_nodes(self) -> bool:
        return self.test.pin_nodes

    def __hash__(self) -> int:
        return hash(self.name + self.test.name + str(self.iterations) + str(self.current_iteration))

    def has_more_iterations(self) -> bool:
        """
        Check if the test has more iterations to run.

        Returns
            bool: True if more iterations are pending, False otherwise.
        """
        return self.current_iteration + 1 < self.iterations

    def increment_step(self) -> int:
        """Advance the trial counter and return the new value."""
        self.step += 1
        return self.step

    @property
    def metric_reporter(self) -> Optional[Type[ReportGenerationStrategy]]:
        if not self.reports:
            return None

        if not self.test.agent_metrics:
            return None

        for r in self.reports:
            if all(metric in r.metrics for metric in self.test.agent_metrics):
                return r

        return None

    def get_metric_value(self, system: System, metric: str) -> MetricValue:
        report = self.metric_reporter
        if report is None:
            return METRIC_ERROR

        return report(system, self).get_metric(metric)

    @property
    def is_dse_job(self) -> bool:
        return self.test.is_dse_job or isinstance(self.num_nodes, list)

    @property
    def is_domain_randomization_active(self) -> bool:
        """
        Whether this run will actually env-sample (domain-randomize) per trial.

        True only when domain randomization is declared (``env_params`` present), the run is a DSE
        job (so a per-trial loop exists - including a ``num_nodes`` sweep), and the agent opts into
        sampling. An unknown agent is treated as opted-in so the dedicated agent-resolution error
        surfaces instead of this one.
        """
        if not self.test.is_domain_randomization_enabled:
            return False

        registry = Registry()
        agent = registry.get_agent(self.test.agent) if registry.has_agent(self.test.agent) else None
        return self.is_dse_job and (agent is None or agent.supports_variable_environment)

    @property
    def nnodes(self) -> int:
        """Type safe getter for num_nodes, should only be used on an unrolled DSE job."""
        if isinstance(self.num_nodes, list):
            raise TypeError("num_nodes is a list, cannot be used as a scalar.")
        return self.num_nodes

    @property
    def param_space(self) -> dict[str, Any]:
        cmd_args_dict = flatten_dict(self.test.cmd_args.model_dump())
        extra_env_vars_dict = self.test.extra_env_vars

        action_space: dict[str, Any] = {
            **{
                key: value
                for key, value in cmd_args_dict.items()
                if isinstance(value, list)
                and not self.test.is_dse_excluded_arg(key)
                and not self.test.is_env_sampled(key)
            },
            **{f"extra_env_vars.{key}": value for key, value in extra_env_vars_dict.items() if isinstance(value, list)},
        }
        if isinstance(self.num_nodes, list):
            action_space["NUM_NODES"] = self.num_nodes

        return action_space

    @property
    def all_combinations(self) -> list[dict[str, Any]]:
        if not self.is_dse_job:
            return []

        param_space: dict[str, Any] = self.param_space
        if not param_space:
            return []

        parameter_values: list[Any] = []
        for _, values in param_space.items():
            parameter_values.append(values)
        action_combinations = list(itertools.product(*parameter_values))

        keys = list(param_space.keys())
        all_combinations = [dict(zip(keys, combination, strict=True)) for combination in action_combinations]

        return all_combinations

    def apply_params_set(self, action: dict[str, Any], env_params: dict[str, Any] | None = None) -> "TestRun":
        tdef = self.test.model_copy(deep=True)

        # RNG runs in the env before this call; applying only concrete values keeps this deterministic.
        # action and env_params target disjoint keys, so a plain merge applies both in one pass.
        full_action = action | (env_params or {})
        for key, value in full_action.items():
            if key.startswith("extra_env_vars."):
                tdef.extra_env_vars[key[len("extra_env_vars.") :]] = value
            elif key == "NUM_NODES":
                tdef.num_nodes = value
            else:
                attrs = key.split(".")
                obj = tdef.cmd_args
                for attr in attrs[:-1]:
                    obj = obj[attr] if isinstance(obj, dict) else getattr(obj, attr)
                if isinstance(obj, dict):
                    obj[attrs[-1]] = value
                else:
                    setattr(obj, attrs[-1], value)

        # env_params is validated at parse time; after the overlay its target cmd_args fields hold
        # concrete scalar draws, so re-validating it here would reject weighted specs. Drop it for
        # this validation-only pass, which exists to validate the applied action values.
        validation_args = tdef.model_dump()
        validation_args.pop("env_params", None)
        type(tdef)(**validation_args)  # trigger validation

        new_tr = copy.deepcopy(self)
        new_tr.test = tdef
        return new_tr
