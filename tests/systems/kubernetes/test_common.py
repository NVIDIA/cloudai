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

from cloudai.core import JsonGenStrategy, TestRun
from cloudai.systems.kubernetes import KubernetesSystem


class MyJsonGenStrategy(JsonGenStrategy):
    def gen_json(self) -> dict:
        return {}


@pytest.mark.parametrize(
    "tname,expected",
    [
        ("simple-name", "simple-name"),
        ("name_with_underscores", "name-with-underscores"),
        ("name.with.dots", "name-with-dots"),
        ("name@with#special$chars", "name-with-special-chars"),
        ("NameWithUpperCase", "namewithuppercase"),
        ("a" * 260, "a" * 253),
        ("---leading-and-trailing---", "leading-and-trailing"),
        ("a" * 250 + "-" * 3 + "b" * 10, "a" * 250),  # ensure no trailing hyphens on truncation
    ],
)
def test_job_name_sanitization(k8s_system: KubernetesSystem, base_tr: TestRun, tname: str, expected: str) -> None:
    base_tr.name = tname
    json_gen = MyJsonGenStrategy(k8s_system, base_tr)
    assert json_gen.sanitize_k8s_job_name(base_tr.name) == expected


def test_job_name_sanitization_raises(k8s_system: KubernetesSystem, base_tr: TestRun) -> None:
    base_tr.name = "!@#$%^&*()"
    json_gen = MyJsonGenStrategy(k8s_system, base_tr)
    with pytest.raises(ValueError):
        json_gen.sanitize_k8s_job_name(base_tr.name)
