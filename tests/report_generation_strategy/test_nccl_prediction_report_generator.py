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

from pathlib import Path

import pytest

from cloudai import GitRepo, PredictorConfig
from cloudai.workloads.nccl_test import NCCLTestDefinition
from cloudai.workloads.nccl_test.prediction_report_generator import NcclTestPredictionReportGenerator
from tests.conftest import MyTestDefinition


@pytest.fixture
def test_definition(tmp_path: Path) -> MyTestDefinition:
    repo = GitRepo(
        url="https://github.com/mock/repo.git",
        commit="mock_commit",
        installed_path=tmp_path / "mock/repo",
    )

    predictor = PredictorConfig(
        git_repo=repo,
        venv_path=tmp_path / "mock/venv",
        project_subpath=None,
        dependencies_from_pyproject=True,
    )

    return MyTestDefinition(
        name="mock_test",
        description="A mock test definition",
        test_template_name="mock_template",
        cmd_args={},
        predictor=predictor,
    )


@pytest.fixture
def generator(test_definition: NCCLTestDefinition, tmp_path: Path) -> NcclTestPredictionReportGenerator:
    return NcclTestPredictionReportGenerator("all_reduce", tmp_path, test_definition)


@pytest.mark.parametrize(
    "stdout_content, expected_gpu, expected_devices, expected_ranks",
    [
        (
            """
            # Rank  0 Group  0 Pid 1000 on node1 device  0 [0xaa] NVIDIA H100
            # Rank  1 Group  0 Pid 1001 on node1 device  1 [0xbb] NVIDIA H100
            """,
            "H100",
            2,
            2,
        ),
        (
            """
            # Rank  0 Group  0 Pid 1000 on node1 device  0 [0xaa] NVIDIA A100
            """,
            "A100",
            1,
            1,
        ),
    ],
)
def test_extract_device_info(
    generator: NcclTestPredictionReportGenerator,
    tmp_path: Path,
    stdout_content: str,
    expected_gpu: str,
    expected_devices: int,
    expected_ranks: int,
) -> None:
    (tmp_path / "stdout.txt").write_text(stdout_content)
    generator.stdout_path = tmp_path / "stdout.txt"

    gpu_type, num_devices, num_ranks = generator._extract_device_info()
    assert gpu_type == expected_gpu
    assert num_devices == expected_devices
    assert num_ranks == expected_ranks


def test_extract_performance_data(generator: NcclTestPredictionReportGenerator, tmp_path: Path) -> None:
    stdout_content = """
    128     32     float     sum      -1    343.8    0.00    0.00      0    24.05    0.01    0.01      0
    256     64     float     sum      -1    96.89    0.00    0.00      0    24.34    0.01    0.02      0
    """
    (tmp_path / "stdout.txt").write_text(stdout_content)
    generator.stdout_path = tmp_path / "stdout.txt"

    df = generator._extract_performance_data("H100", 8, 16)
    assert not df.empty
    assert set(df.columns) == {"gpu_type", "num_devices_per_node", "num_ranks", "message_size", "measured_dur"}
