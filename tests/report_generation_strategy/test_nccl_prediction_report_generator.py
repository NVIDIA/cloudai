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
from unittest.mock import Mock

import pandas as pd
import pytest

from cloudai import GitRepo, PredictorConfig, Test, TestRun
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nccl_test import NCCLCmdArgs
from cloudai.workloads.nccl_test.prediction_report_generation_strategy import NcclTestPredictionReportGenerationStrategy
from cloudai.workloads.nccl_test.predictor.default import DefaultNCCLTestPredictor
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

    cmd_args = NCCLCmdArgs(subtest_name="all_reduce_perf")

    return MyTestDefinition(
        name="mock_test",
        description="A mock test definition",
        test_template_name="mock_template",
        cmd_args=cmd_args,
        predictor=predictor,
    )


@pytest.fixture
def generator(
    slurm_system: SlurmSystem, test_definition: MyTestDefinition, tmp_path: Path
) -> NcclTestPredictionReportGenerationStrategy:
    test = Test(
        test_definition=test_definition,
        test_template=Mock(),
    )
    test_run = TestRun(name="mock_test_run", test=test, num_nodes=1, nodes=[], output_path=tmp_path)
    return NcclTestPredictionReportGenerationStrategy(slurm_system, test_run)


def test_extract_performance_data(tmp_path: Path) -> None:
    csv_report_path = tmp_path / "cloudai_nccl_test_csv_report.csv"

    mock_csv_data = pd.DataFrame(
        {
            "GPU Type": ["H100", "H100"],
            "Devices per Node": [8, 8],
            "Ranks": [16, 16],
            "Size (B)": [128, 256],
            "Time (us) Out-of-place": [343.8, 96.89],
        }
    )

    mock_csv_data.to_csv(csv_report_path, index=False)

    predictor = DefaultNCCLTestPredictor()

    df = pd.read_csv(csv_report_path)

    df = predictor._extract_performance_data(df)

    assert not df.empty
    assert set(df.columns) == {"GPU Type", "Devices per Node", "Ranks", "message_size", "measured_dur"}
    assert df.iloc[0]["GPU Type"] == "H100"
    assert df.iloc[0]["Devices per Node"] == 8
    assert df.iloc[0]["Ranks"] == 16
    assert df.iloc[0]["message_size"] == 128
    assert df.iloc[0]["measured_dur"] == 343.8
    assert df.iloc[1]["message_size"] == 256
    assert df.iloc[1]["measured_dur"] == 96.89
