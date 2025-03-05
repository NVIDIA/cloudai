# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai import Test, TestRun
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.megatron_run import (
    CheckpointTimingReportGenerationStrategy,
    MegatronRunCmdArgs,
    MegatronRunTestDefinition,
)


@pytest.fixture
def megatron_tr(tmp_path: Path) -> TestRun:
    test = Test(
        test_definition=MegatronRunTestDefinition(
            name="megatron_run",
            description="desc",
            test_template_name="t",
            cmd_args=MegatronRunCmdArgs(docker_image_url="http://url", run_script=Path(__file__)),
        ),
        test_template=Mock(),
    )
    tr = TestRun(name="n", test=test, num_nodes=1, nodes=[], output_path=tmp_path)

    stdout_content = """
        save-checkpoint ................................: (1.1, 10.2)
    load-checkpoint ................................: (2.1, 22.1)
        save-checkpoint ................................: (1000.1, 10000.2)
    """
    (tr.output_path / "stdout.txt").write_text(stdout_content)

    return tr


def test_checkpoint_timings_reporter(slurm_system: SlurmSystem, megatron_tr: TestRun) -> None:
    test_dir = megatron_tr.output_path

    strategy = CheckpointTimingReportGenerationStrategy(slurm_system, megatron_tr)
    assert strategy.can_handle_directory() is True

    strategy.generate_report()

    csv_report_path = test_dir / "report.csv"
    assert csv_report_path.is_file(), "CSV report was not generated."

    df = pd.read_csv(csv_report_path)
    assert not df.empty, "CSV report is empty."

    assert df.shape == (3, 3), "CSV report has incorrect shape."
    assert df.columns.tolist() == ["checkpoint_type", "min", "max"], "CSV report has incorrect columns."

    assert df["checkpoint_type"].tolist() == ["save", "save", "load"], "CSV report has incorrect checkpoint types."
    assert df["min"].tolist() == [1.1, 1000.1, 2.1], "CSV report has incorrect min values."
    assert df["max"].tolist() == [10.2, 10000.2, 22.1], "CSV report has incorrect max values."
