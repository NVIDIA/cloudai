# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import cast

import pytest
from pydantic import ValidationError

import cloudai.metrics
from cloudai.models.scenario import TestRunModel, TestScenarioModel


def test_sol_is_validated_by_metric_schema() -> None:
    sol = cloudai.metrics.parse_sol_spec(
        {
            "transfer_bandwidth": {"read": 100, "default": 80},
            "collective_bus_bandwidth": {"all_reduce": {"in_place": 350}},
        }
    )

    assert cast(cloudai.metrics.TransferSOL, sol["transfer_bandwidth"]).read == 100
    assert cast(cloudai.metrics.CollectiveSOL, sol["collective_bus_bandwidth"]).root["all_reduce"].in_place == 350


@pytest.mark.parametrize(
    "spec",
    [
        {"not_a_metric": {"default": 1}},
        {"transfer_bandwidth": {"backend": "UCX"}},
        {"transfer_bandwidth": {"read": -1}},
    ],
)
def test_invalid_sol_fails_during_config_parsing(spec: dict) -> None:
    with pytest.raises((ValueError, ValidationError)):
        cloudai.metrics.parse_sol_spec(spec)


def test_attainment_respects_metric_optimization_direction() -> None:
    config = cloudai.metrics.parse_sol_spec(
        {
            "transfer_bandwidth": {"default": 100},
            "transfer_latency": {"default": 10},
        }
    )
    coordinates = cloudai.metrics.TransferCoordinates(payload_size_bytes=1024)

    bandwidth = cloudai.metrics.assess_observation(
        cloudai.metrics.MetricObservation(cloudai.metrics.TRANSFER_BANDWIDTH, 80, coordinates), config
    )
    latency = cloudai.metrics.assess_observation(
        cloudai.metrics.MetricObservation(cloudai.metrics.TRANSFER_LATENCY, 12.5, coordinates), config
    )

    assert bandwidth.attainment == pytest.approx(0.8)
    assert latency.attainment == pytest.approx(0.8)


def test_collective_sol_uses_collective_and_placement() -> None:
    config = cloudai.metrics.parse_sol_spec(
        {"collective_bus_bandwidth": {"all_reduce": {"in_place": 300, "out_of_place": 250}}}
    )
    observation = cloudai.metrics.MetricObservation(
        cloudai.metrics.COLLECTIVE_BUS_BANDWIDTH,
        240,
        cloudai.metrics.CollectiveCoordinates(
            collective="all_reduce", placement="out_of_place", message_size_bytes=1024
        ),
    )

    assessment = cloudai.metrics.assess_observation(observation, config)

    assert assessment.sol == 250
    assert assessment.attainment == pytest.approx(0.96)


def test_scenario_and_test_case_accept_structured_sol() -> None:
    scenario = TestScenarioModel.model_validate(
        {
            "name": "example",
            "sol": {"transfer_bandwidth": {"default": 80}},
            "Tests": [
                {
                    "id": "case",
                    "test_name": "nixl",
                    "sol": {"transfer_bandwidth": {"read": 100}},
                }
            ],
        }
    )

    assert cast(cloudai.metrics.TransferSOL, scenario.sol["transfer_bandwidth"]).default == 80
    assert isinstance(scenario.tests[0], TestRunModel)
    test_sol = scenario.tests[0].sol
    assert isinstance(test_sol, dict)
    assert cast(cloudai.metrics.TransferSOL, test_sol["transfer_bandwidth"]).read == 100


def test_sol_precedence_replaces_a_metric_at_the_more_specific_level() -> None:
    system = cloudai.metrics.parse_sol_spec(
        {"transfer_bandwidth": {"default": 80}, "transfer_latency": {"default": 10}}
    )
    scenario = cloudai.metrics.parse_sol_spec({"transfer_bandwidth": {"default": 90}})
    test_case = cloudai.metrics.parse_sol_spec({"transfer_bandwidth": {"read": 100}})

    merged = cloudai.metrics.merge_sol_configs(system, scenario, test_case)

    assert cast(cloudai.metrics.TransferSOL, merged["transfer_bandwidth"]).read == 100
    assert cast(cloudai.metrics.TransferSOL, merged["transfer_bandwidth"]).default is None
    assert cast(cloudai.metrics.TransferSOL, merged["transfer_latency"]).default == 10
