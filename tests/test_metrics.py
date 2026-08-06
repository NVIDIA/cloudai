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

import pytest
import toml
from pydantic import ValidationError

import cloudai.metrics
from cloudai.models.scenario import TestRunModel, TestScenarioModel


def test_sol_is_validated_by_metric_schema() -> None:
    sol = cloudai.metrics.parse_sol_spec(
        {
            "bandwidth": [
                {"value": 80},
                {"value": 100, "match": {"operation": "read", "bandwidth_basis": "payload"}},
            ]
        }
    )

    assert sol["bandwidth"][0].value == 80
    assert sol["bandwidth"][0].match == {}
    assert sol["bandwidth"][1].match == {"operation": "read", "bandwidth_basis": "payload"}


def test_sol_target_toml_syntax() -> None:
    config = toml.loads(
        """
        [[sol.bandwidth]]
        value = 80

        [[sol.bandwidth]]
        value = 100
        match = { operation = "read" }
        """
    )

    parsed = cloudai.metrics.parse_sol_spec(config["sol"])

    targets = parsed["bandwidth"]
    assert [target.value for target in targets] == [80, 100]
    assert targets[1].match == {"operation": "read"}


@pytest.mark.parametrize(
    "spec",
    [
        {"not_a_metric": [{"value": 1}]},
        {"bandwidth": [{"value": 1, "match": {"not_a_dimension": "UCX"}}]},
        {"bandwidth": [{"value": -1}]},
        {"bandwidth": {"value": 1}},
        {"bandwidth": []},
    ],
)
def test_invalid_sol_fails_during_config_parsing(spec: dict) -> None:
    with pytest.raises((ValueError, ValidationError)):
        cloudai.metrics.parse_sol_spec(spec)


def test_attainment_respects_metric_optimization_direction() -> None:
    config = cloudai.metrics.parse_sol_spec(
        {
            "bandwidth": [{"value": 100}],
            "latency": [{"value": 10}],
        }
    )
    dimensions = {"size_bytes": 1024}

    bandwidth = cloudai.metrics.assess_observation(
        cloudai.metrics.MetricObservation(cloudai.metrics.BANDWIDTH, 80, dimensions), config
    )
    latency = cloudai.metrics.assess_observation(
        cloudai.metrics.MetricObservation(cloudai.metrics.LATENCY, 12.5, dimensions), config
    )

    assert bandwidth.attainment == pytest.approx(0.8)
    assert latency.attainment == pytest.approx(0.8)


def test_one_bandwidth_metric_supports_collective_and_point_to_point_targets() -> None:
    config = cloudai.metrics.parse_sol_spec(
        {
            "bandwidth": [
                {
                    "value": 250,
                    "match": {
                        "operation": "all_reduce",
                        "placement": "out_of_place",
                        "bandwidth_basis": "bus",
                    },
                },
                {
                    "value": 100,
                    "match": {"operation": "write", "bandwidth_basis": "payload"},
                },
            ]
        }
    )
    collective = cloudai.metrics.MetricObservation(
        cloudai.metrics.BANDWIDTH,
        240,
        {
            "operation": "all_reduce",
            "placement": "out_of_place",
            "size_bytes": 1024,
            "bandwidth_basis": "bus",
        },
    )
    point_to_point = cloudai.metrics.MetricObservation(
        cloudai.metrics.BANDWIDTH,
        80,
        {"operation": "write", "size_bytes": 1024, "bandwidth_basis": "payload"},
    )

    collective_assessment = cloudai.metrics.assess_observation(collective, config)
    point_to_point_assessment = cloudai.metrics.assess_observation(point_to_point, config)

    assert collective_assessment.sol == 250
    assert collective_assessment.attainment == pytest.approx(0.96)
    assert point_to_point_assessment.sol == 100
    assert point_to_point_assessment.attainment == pytest.approx(0.8)


def test_collective_sol_accepts_zero_byte_nccl_observation() -> None:
    config = cloudai.metrics.parse_sol_spec(
        {
            "latency": [
                {
                    "value": 10,
                    "match": {
                        "operation": "all_gather",
                        "placement": "out_of_place",
                        "size_bytes": 0,
                    },
                }
            ]
        }
    )
    observation = cloudai.metrics.MetricObservation(
        cloudai.metrics.LATENCY,
        12.5,
        {"operation": "all_gather", "placement": "out_of_place", "size_bytes": 0},
    )

    assessment = cloudai.metrics.assess_observation(observation, config)

    assert assessment.sol == 10
    assert assessment.attainment == pytest.approx(0.8)


def test_scenario_and_test_case_accept_structured_sol() -> None:
    scenario = TestScenarioModel.model_validate(
        {
            "name": "example",
            "sol": {"bandwidth": [{"value": 80}]},
            "Tests": [
                {
                    "id": "case",
                    "test_name": "nixl",
                    "sol": {"bandwidth": [{"value": 100, "match": {"operation": "read"}}]},
                }
            ],
        }
    )

    scenario_targets = scenario.sol["bandwidth"]
    assert scenario_targets[0].value == 80
    assert scenario_targets[0].match == {}
    assert isinstance(scenario.tests[0], TestRunModel)
    test_sol = scenario.tests[0].sol
    assert isinstance(test_sol, dict)
    test_targets = test_sol["bandwidth"]
    assert test_targets[0].match == {"operation": "read"}
    assert scenario.model_dump()["tests"][0]["sol"]["bandwidth"][0]["match"] == {"operation": "read"}


def test_sol_precedence_replaces_a_metric_at_the_more_specific_level() -> None:
    system = cloudai.metrics.parse_sol_spec({"bandwidth": [{"value": 80}], "latency": [{"value": 10}]})
    scenario = cloudai.metrics.parse_sol_spec({"bandwidth": [{"value": 90}]})
    test_case = cloudai.metrics.parse_sol_spec({"bandwidth": [{"value": 100, "match": {"operation": "read"}}]})

    merged = cloudai.metrics.merge_sol_configs(system, scenario, test_case)

    bandwidth_targets = merged["bandwidth"]
    latency_targets = merged["latency"]
    assert len(bandwidth_targets) == 1
    assert bandwidth_targets[0].value == 100
    assert bandwidth_targets[0].match == {"operation": "read"}
    assert latency_targets[0].value == 10


def test_most_specific_matching_sol_target_wins() -> None:
    config = cloudai.metrics.parse_sol_spec(
        {
            "bandwidth": [
                {"value": 80},
                {"value": 100, "match": {"operation": "read"}},
                {"value": 120, "match": {"operation": "read", "size_bytes": 1024}},
            ]
        }
    )
    observation = cloudai.metrics.MetricObservation(
        cloudai.metrics.BANDWIDTH,
        90,
        {"operation": "read", "size_bytes": 1024},
    )

    assessment = cloudai.metrics.assess_observation(observation, config)

    assert assessment.sol == 120
    assert assessment.target is not None
    assert assessment.target.match == {"operation": "read", "size_bytes": 1024}
    assert assessment.attainment == pytest.approx(0.75)


def test_metric_view_infers_size_axis_and_placement_series() -> None:
    assessments = [
        cloudai.metrics.assess_observation(
            cloudai.metrics.MetricObservation(
                cloudai.metrics.BANDWIDTH,
                100,
                {
                    "operation": "all_gather",
                    "placement": placement,
                    "size_bytes": size,
                    "bandwidth_basis": "bus",
                },
            ),
            {},
        )
        for size in (1024, 2048)
        for placement in ("in_place", "out_of_place")
    ]

    view = cloudai.metrics.build_metric_view(cloudai.metrics.BANDWIDTH, [assessments])

    assert view is not None
    assert view.x_dimension == "size_bytes"
    assert view.series_dimensions == ("placement",)


def test_metric_view_keeps_scalar_metric_table_only() -> None:
    assessment = cloudai.metrics.assess_observation(
        cloudai.metrics.MetricObservation(cloudai.metrics.LATENCY, 10, {"operation": "write"}),
        {},
    )

    view = cloudai.metrics.build_metric_view(cloudai.metrics.LATENCY, [[assessment]])

    assert view is not None
    assert view.x_dimension is None
    assert view.series_dimensions == ()


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
        cloudai.metrics.parse_sol_spec({"bandwidth": targets})
