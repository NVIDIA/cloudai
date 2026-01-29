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

import math

import pytest

from cloudai.workloads.ai_dynamo.calc_percentile_csv import compute_percentile, parse_float_safe, summarize


def test_compute_percentile_empty():
    assert math.isnan(compute_percentile([], 50))


def test_compute_percentile_single_value():
    assert compute_percentile([5.0], 50) == 5.0
    assert compute_percentile([5.0], 0) == 5.0
    assert compute_percentile([5.0], 100) == 5.0


def test_compute_percentile_multiple_values():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert compute_percentile(values, 0) == 1.0
    assert compute_percentile(values, 50) == 3.0
    assert compute_percentile(values, 100) == 5.0


def test_compute_percentile_interpolation():
    values = [1.0, 2.0, 3.0, 4.0]
    # Should interpolate between values
    result = compute_percentile(values, 50)
    assert 2.0 <= result <= 3.0


def test_parse_float_safe_valid():
    assert parse_float_safe("3.14") == 3.14
    assert parse_float_safe(42) == 42.0
    assert parse_float_safe(3.14) == 3.14


def test_parse_float_safe_invalid():
    assert math.isnan(parse_float_safe("invalid"))
    assert math.isnan(parse_float_safe(None))
    assert math.isnan(parse_float_safe(""))


def test_summarize_empty():
    result = summarize([])
    assert math.isnan(result["avg"])
    assert math.isnan(result["min"])
    assert math.isnan(result["max"])
    assert math.isnan(result["p50"])


def test_summarize_single_value():
    result = summarize([10.0])
    assert result["avg"] == 10.0
    assert result["min"] == 10.0
    assert result["max"] == 10.0
    assert result["p50"] == 10.0


def test_summarize_multiple_values():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = summarize(values)
    assert result["avg"] == 3.0
    assert result["min"] == 1.0
    assert result["max"] == 5.0
    assert result["p50"] == 3.0
    assert result["p25"] == 2.0
    assert result["p75"] == 4.0


def test_summarize_percentiles():
    values = [float(x) for x in range(1, 101)]  # 1 to 100
    result = summarize(values)
    assert result["p1"] == pytest.approx(1.99, abs=0.1)
    assert result["p99"] == pytest.approx(99.01, abs=0.1)
    assert result["p50"] == pytest.approx(50.5, abs=0.1)
