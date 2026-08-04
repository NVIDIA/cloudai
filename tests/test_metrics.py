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
import toml
from pydantic import ValidationError

import cloudai.metrics
from cloudai.models.scenario import TestRunModel, TestScenarioModel


def test_sol_is_validated_by_metric_schema() -> None:
    sol = cloudai.metrics.parse_sol_spec(
        {
            "transfer_bandwidth": [
                {"value": 80},
                {"value": 100, "match": {"operation": "read"}},
            ],
            "collective_bus_bandwidth": [
                {"value": 350, "match": {"collective": "all_reduce", "placement": "in_place"}}
            ],
        }
    )

    transfer_targets = cast(list[cloudai.metrics.TransferSOLTarget], sol["transfer_bandwidth"])
    collective_targets = cast(list[cloudai.metrics.CollectiveSOLTarget], sol["collective_bus_bandwidth"])
    assert transfer_targets[0].value == 80
    assert transfer_targets[0].selector() == {}
    assert transfer_targets[1].selector() == {"operation": "read"}
    assert collective_targets[0].selector() == {"collective": "all_reduce", "placement": "in_place"}


def test_sol_target_toml_syntax() -> None:
    config = toml.loads(
        """
        [[sol.transfer_bandwidth]]
        value = 80

        [[sol.transfer_bandwidth]]
        value = 100
        match = { operation = "read" }
        """
    )

    parsed = cloudai.metrics.parse_sol_spec(config["sol"])

    targets = cast(list[cloudai.metrics.TransferSOLTarget], parsed["transfer_bandwidth"])
    assert [target.value for target in targets] == [80, 100]
    assert targets[1].selector() == {"operation": "read"}


@pytest.mark.parametrize(
    "spec",
    [
        {"not_a_metric": [{"value": 1}]},
        {"transfer_bandwidth": [{"value": 1, "match": {"backend": "UCX"}}]},
        {"transfer_bandwidth": [{"value": -1}]},
        {"transfer_bandwidth": {"value": 1}},
        {"transfer_bandwidth": []},
    ],
)
def test_invalid_sol_fails_during_config_parsing(spec: dict) -> None:
    with pytest.raises((ValueError, ValidationError)):
        cloudai.metrics.parse_sol_spec(spec)


def test_attainment_respects_metric_optimization_direction() -> None:
    config = cloudai.metrics.parse_sol_spec(
        {
            "transfer_bandwidth": [{"value": 100}],
            "transfer_latency": [{"value": 10}],
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
        {
            "collective_bus_bandwidth": [
                {"value": 300, "match": {"collective": "all_reduce", "placement": "in_place"}},
                {"value": 250, "match": {"collective": "all_reduce", "placement": "out_of_place"}},
            ]
        }
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


def test_collective_sol_accepts_zero_byte_nccl_observation() -> None:
    config = cloudai.metrics.parse_sol_spec(
        {
            "collective_latency": [
                {
                    "value": 10,
                    "match": {
                        "collective": "all_gather",
                        "placement": "out_of_place",
                        "message_size_bytes": 0,
                    },
                }
            ]
        }
    )
    observation = cloudai.metrics.MetricObservation(
        cloudai.metrics.COLLECTIVE_LATENCY,
        12.5,
        cloudai.metrics.CollectiveCoordinates(collective="all_gather", placement="out_of_place", message_size_bytes=0),
    )

    assessment = cloudai.metrics.assess_observation(observation, config)

    assert assessment.sol == 10
    assert assessment.attainment == pytest.approx(0.8)


def test_scenario_and_test_case_accept_structured_sol() -> None:
    scenario = TestScenarioModel.model_validate(
        {
            "name": "example",
            "sol": {"transfer_bandwidth": [{"value": 80}]},
            "Tests": [
                {
                    "id": "case",
                    "test_name": "nixl",
                    "sol": {"transfer_bandwidth": [{"value": 100, "match": {"operation": "read"}}]},
                }
            ],
        }
    )

    scenario_targets = cast(list[cloudai.metrics.TransferSOLTarget], scenario.sol["transfer_bandwidth"])
    assert scenario_targets[0].value == 80
    assert scenario_targets[0].selector() == {}
    assert isinstance(scenario.tests[0], TestRunModel)
    test_sol = scenario.tests[0].sol
    assert isinstance(test_sol, dict)
    test_targets = cast(list[cloudai.metrics.TransferSOLTarget], test_sol["transfer_bandwidth"])
    assert test_targets[0].selector() == {"operation": "read"}
    assert scenario.model_dump()["tests"][0]["sol"]["transfer_bandwidth"][0]["match"] == {
        "operation": "read",
        "payload_size_bytes": None,
        "batch_size": None,
    }


def test_sol_precedence_replaces_a_metric_at_the_more_specific_level() -> None:
    system = cloudai.metrics.parse_sol_spec(
        {"transfer_bandwidth": [{"value": 80}], "transfer_latency": [{"value": 10}]}
    )
    scenario = cloudai.metrics.parse_sol_spec({"transfer_bandwidth": [{"value": 90}]})
    test_case = cloudai.metrics.parse_sol_spec({"transfer_bandwidth": [{"value": 100, "match": {"operation": "read"}}]})

    merged = cloudai.metrics.merge_sol_configs(system, scenario, test_case)

    bandwidth_targets = cast(list[cloudai.metrics.TransferSOLTarget], merged["transfer_bandwidth"])
    latency_targets = cast(list[cloudai.metrics.TransferSOLTarget], merged["transfer_latency"])
    assert len(bandwidth_targets) == 1
    assert bandwidth_targets[0].value == 100
    assert bandwidth_targets[0].selector() == {"operation": "read"}
    assert latency_targets[0].value == 10


def test_most_specific_matching_sol_target_wins() -> None:
    config = cloudai.metrics.parse_sol_spec(
        {
            "transfer_bandwidth": [
                {"value": 80},
                {"value": 100, "match": {"operation": "read"}},
                {"value": 120, "match": {"operation": "read", "payload_size_bytes": 1024}},
            ]
        }
    )
    observation = cloudai.metrics.MetricObservation(
        cloudai.metrics.TRANSFER_BANDWIDTH,
        90,
        cloudai.metrics.TransferCoordinates(operation="read", payload_size_bytes=1024),
    )

    assessment = cloudai.metrics.assess_observation(observation, config)

    assert assessment.sol == 120
    assert assessment.attainment == pytest.approx(0.75)


@pytest.mark.parametrize(
    "targets",
    [
        [{"value": 80}, {"value": 90}],
        [
            {"value": 100, "match": {"operation": "read"}},
            {"value": 110, "match": {"batch_size": 1}},
        ],
    ],
)
def test_ambiguous_sol_targets_fail_during_config_parsing(targets: list[dict]) -> None:
    with pytest.raises(ValueError, match="Ambiguous SOL targets"):
        cloudai.metrics.parse_sol_spec({"transfer_bandwidth": targets})
