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

from types import SimpleNamespace

import pytest

import cloudai.metrics


def test_parse_sol_spec() -> None:
    config = cloudai.metrics.parse_sol_spec(
        {
            "bandwidth": [
                {"value": 100},
                {"value": 80, "match": {"operation": "write", "size_bytes": 4096}},
            ]
        }
    )

    assert [target.model_dump() for target in config["bandwidth"]] == [
        {"value": 100.0, "match": {}},
        {"value": 80.0, "match": {"operation": "write", "size_bytes": 4096}},
    ]


@pytest.mark.parametrize(
    "value,error",
    [
        ({"throughput": [{"value": 100}]}, "Unknown SOL metric"),
        ({"bandwidth": []}, "non-empty list"),
        ({"bandwidth": [{"value": 0}]}, "greater than 0"),
        ({"bandwidth": [{"value": 100, "match": {"placement": "sideways"}}]}, "in_place"),
        (
            {
                "bandwidth": [
                    {"value": 100, "match": {"operation": "write"}},
                    {"value": 80, "match": {"backend": "ucx"}},
                ]
            },
            "Ambiguous SOL targets",
        ),
    ],
)
def test_parse_sol_spec_rejects_invalid_config(value: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        cloudai.metrics.parse_sol_spec(value)


def test_merge_sol_configs() -> None:
    system = cloudai.metrics.parse_sol_spec(
        {
            "bandwidth": [{"value": 100}],
            "latency": [{"value": 10}],
        }
    )
    test = cloudai.metrics.parse_sol_spec({"bandwidth": [{"value": 120, "match": {"operation": "write"}}]})

    merged = cloudai.metrics.merge_sol_configs(None, system, test)

    assert merged == {
        "bandwidth": test["bandwidth"],
        "latency": system["latency"],
    }


@pytest.mark.parametrize("structured", [False, True])
def test_metric_assessment_only_uses_metric_keyed_targets(structured: bool) -> None:
    observations = [
        cloudai.metrics.MetricObservation(cloudai.metrics.BANDWIDTH, 50, {}),
        cloudai.metrics.MetricObservation(cloudai.metrics.LATENCY, 5, {}),
    ]
    sol = cloudai.metrics.SOLMetricModel.model_validate({"bandwidth": [{"value": 100}], "latency": [{"value": 10}]})
    run = SimpleNamespace(
        test=SimpleNamespace(metric_observations=lambda system, run: observations), sol=sol if structured else 100.0
    )
    assessments = cloudai.metrics.assess_test_run_metrics(None, run)
    assert [assessment.sol for assessment in assessments] == ([100, 10] if structured else [None, None])
