#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict

import pytest
from cloudai.parser.system_parser.slurm_system_parser import SlurmSystemParser
from cloudai.systems.slurm import SlurmSystem


@pytest.fixture
def example_data() -> Dict[str, Any]:
    return {
        "name": "test_system",
        "install_path": "/fake/path",
        "output_path": "/fake/output",
        "default_partition": "main",
        "partitions": {
            "main": {
                "name": "main",
                "nodes": ["node-[033-034]"],
                "groups": {"group1": {"name": "group1", "nodes": ["node-033"]}},
            },
            "backup": {
                "name": "backup",
                "nodes": ["node-[01-02]"],
                "groups": {"group2": {"name": "group2", "nodes": ["node-01"]}},
            },
        },
        "cache_docker_images_locally": "True",
    }


def test_parse_slurm_system_parser(example_data):
    parser = SlurmSystemParser()
    slurm_system = parser.parse(example_data)

    assert isinstance(slurm_system, SlurmSystem)
    assert slurm_system.name == "test_system"
    assert slurm_system.install_path == "/fake/path"
    assert slurm_system.output_path == "/fake/output"
    assert slurm_system.default_partition == "main"
    assert slurm_system.cache_docker_images_locally is True
    assert "main" in slurm_system.partitions
    assert "backup" in slurm_system.partitions
    assert "group1" in slurm_system.groups["main"]
    assert "group2" in slurm_system.groups["backup"]


@pytest.mark.parametrize(
    "input_value, expected_result",
    [
        ("True", True),
        ("False", False),
        ("true", True),
        ("false", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False),
        (True, True),
        (False, False),
    ],
)
def test_str_to_bool_conversion(input_value, expected_result):
    parser = SlurmSystemParser()
    result = parser.parse(
        {
            "name": "test_system",
            "install_path": "/fake/path",
            "output_path": "/fake/output",
            "default_partition": "main",
            "partitions": {
                "main": {
                    "name": "main",
                    "nodes": ["node-[033-034]"],
                    "groups": {"group1": {"name": "group1", "nodes": ["node-033"]}},
                },
                "backup": {
                    "name": "backup",
                    "nodes": ["node-[01-02]"],
                    "groups": {"group2": {"name": "group2", "nodes": ["node-01"]}},
                },
            },
            "cache_docker_images_locally": input_value,
        }
    )

    assert result.cache_docker_images_locally == expected_result
