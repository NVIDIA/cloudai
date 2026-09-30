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

from datetime import timedelta

from cloudai.util.utils import parse_time_limit


def test_two_field_slurm_time_is_minutes_and_seconds():
    assert parse_time_limit("1:30") == timedelta(minutes=1, seconds=30)
    assert parse_time_limit("0:30") == timedelta(seconds=30)
    assert parse_time_limit("90:00") == timedelta(minutes=90)


def test_other_slurm_time_forms_stay():
    assert parse_time_limit("1:30:00") == timedelta(hours=1, minutes=30)
    assert parse_time_limit("00:20:00") == timedelta(minutes=20)
    assert parse_time_limit("30m") == timedelta(minutes=30)
    assert parse_time_limit("1-00:00:00") == timedelta(days=1)
