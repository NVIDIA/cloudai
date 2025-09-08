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

import pytest

from cloudai.core import Test, TestRun, TestTemplate
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.nixl_bench import NIXLBenchCmdArgs, NIXLBenchTestDefinition
from cloudai.workloads.nixl_bench.nixl_bench import extract_nixl_data

LEGACY_FORMAT = """
Block Size (B)      Batch Size     Avg Lat. (us)  B/W (MiB/Sec)  B/W (GiB/Sec)  B/W (GB/Sec)
--------------------------------------------------------------------------------
4096                1              21.1726        184.495        0.180171       0.193457
8192                1              21.7391        359.376        0.350953       0.376833
33554432            1              24.6508        1.29813e+06    1267.71        1361.19
67108864            1              39.746         1.61022e+06    1572.48        1688.44
"""

NEW_FORMAT = """
Block Size (B)      Batch Size     B/W (GB/Sec)   Avg Lat. (us)  Avg Prep (us)  P99 Prep (us)  Avg Post (us)  P99 Post (us)  Avg Tx (us)    P99 Tx (us)
----------------------------------------------------------------------------------------------------------------------------------------------------------------
4096                1              0.958841       4.3            11.0           11.0           0.8            1.0            3.5            4.0
8192                1              1.745779       4.7            13.0           13.0           0.8            1.0            3.8            5.0
33554432            1              23.506385      1427.5         13.0           13.0           1.0            9.0            1426.0         1446.0
67108864            1              23.582432      2845.7         13.0           13.0           0.9            9.0            2844.5         2851.0
"""  # noqa: E501


@pytest.fixture
def nixl_tr(tmp_path: Path, slurm_system: SlurmSystem) -> TestRun:
    test = Test(
        test_definition=NIXLBenchTestDefinition(
            name="nixl",
            description="desc",
            test_template_name="t",
            cmd_args=NIXLBenchCmdArgs(docker_image_url="fake://url/nixl", path_to_benchmark="fake://url/nixl_bench"),
        ),
        test_template=TestTemplate(system=slurm_system),
    )
    tr = TestRun(name="nixl_test", test=test, num_nodes=2, nodes=[], output_path=tmp_path)
    return tr


@pytest.mark.parametrize(
    "sample,exp_latency,exp_bw",
    [
        (LEGACY_FORMAT, [21.1726, 21.7391, 24.6508, 39.746], [0.193457, 0.376833, 1361.19, 1688.44]),
        (NEW_FORMAT, [4.3, 4.7, 1427.5, 2845.7], [0.958841, 1.745779, 23.506385, 23.582432]),
    ],
    ids=["LegacyFormat", "NewFormat"],
)
def test_nixl_bench_report_parsing(tmp_path: Path, sample: str, exp_latency: list[float], exp_bw: list[float]):
    (tmp_path / "nixl_bench.log").write_text(sample)
    df = extract_nixl_data(tmp_path / "nixl_bench.log")
    assert df.shape == (4, 4)
    assert df["block_size"].tolist() == [4096, 8192, 33554432, 67108864]
    assert df["batch_size"].tolist() == [1, 1, 1, 1]
    assert df["avg_lat"].tolist() == exp_latency
    assert df["bw_gb_sec"].tolist() == exp_bw


class TestWasRunSuccessful:
    def test_no_file(self, nixl_tr: TestRun):
        assert not nixl_tr.test.test_definition.was_run_successful(nixl_tr).is_successful

    def test_no_data(self, nixl_tr: TestRun):
        nixl_tr.output_path.mkdir(parents=True, exist_ok=True)
        nixl_tr.output_path.joinpath("stdout.txt").write_text("")
        assert not nixl_tr.test.test_definition.was_run_successful(nixl_tr).is_successful

    @pytest.mark.parametrize("sample", [LEGACY_FORMAT, NEW_FORMAT], ids=["LegacyFormat", "NewFormat"])
    def test_was_run_successful(self, nixl_tr: TestRun, sample: str):
        nixl_tr.output_path.mkdir(parents=True, exist_ok=True)
        nixl_tr.output_path.joinpath("stdout.txt").write_text(sample)
        assert nixl_tr.test.test_definition.was_run_successful(nixl_tr).is_successful
