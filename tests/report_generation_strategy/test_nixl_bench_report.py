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

from cloudai.core import Test, TestRun
from cloudai.workloads.common.nixl import extract_nixlbench_data
from cloudai.workloads.nixl_bench import NIXLBenchCmdArgs, NIXLBenchTestDefinition

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
def nixl_tr(tmp_path: Path) -> TestRun:
    test = Test(
        test_definition=NIXLBenchTestDefinition(
            name="nixl",
            description="desc",
            test_template_name="t",
            cmd_args=NIXLBenchCmdArgs(docker_image_url="fake://url/nixl", path_to_benchmark="fake://url/nixl_bench"),
        )
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
    df = extract_nixlbench_data(tmp_path / "nixl_bench.log")
    assert df.shape == (4, 4)
    assert df["block_size"].tolist() == [4096, 8192, 33554432, 67108864]
    assert df["batch_size"].tolist() == [1, 1, 1, 1]
    assert df["avg_lat"].tolist() == exp_latency
    assert df["bw_gb_sec"].tolist() == exp_bw


def test_nixlbench_report_parsing__noisy_output(tmp_path: Path):
    sample = """
----------------------------------------------------------------------------------------------------------------------------------------------------------------

Block Size (B)      Batch Size     B/W (GB/Sec)   Avg Lat. (us)  Avg Prep (us)  P99 Prep (us)  Avg Post (us)  P99 Post (us)  Avg Tx (us)    P99 Tx (us)
----------------------------------------------------------------------------------------------------------------------------------------------------------------
[1757492049.543051] [ptyche0321:627036:0]   +--------------------------------+---------------------------------------------------------+
[1757492049.543052] [ptyche0321:627036:0]   | ucp_context_0 intra-node cfg#2 | active message by ucp_am_send*(egr) from host memory    |
[1757492049.543053] [ptyche0321:627036:0]   +--------------------------------+-------------------------------------+-------------------+
[1757492049.543053] [ptyche0321:627036:0]   |                        0..8184 | short                               | tcp/enP6p3s0f1np1 |
[1757492049.543053] [ptyche0321:627036:0]   |                      8185..inf | multi-frag copy-in                  | tcp/enP6p3s0f1np1 |
[1757492049.543054] [ptyche0321:627036:0]   +--------------------------------+-------------------------------------+-------------------+
4096                1              0.198413       20.6           13.0           13.0           10.6           16.0           10.1           14.0
8192                1              0.381410       21.5           9.0            9.0            10.9           17.0           10.6           15.0
16384               1              0.786957       20.8           7.0            7.0            10.5           15.0           10.3           16.0
32768               1              1.580617       20.7           8.0            8.0            10.6           13.0           10.1           15.0
65536               1              3.151882       20.8           6.0            6.0            10.5           16.0           10.3           15.0
131072              1              6.133162       21.4           6.0            6.0            10.9           14.0           10.5           16.0
262144              1              12.458916      21.0           12.0           12.0           10.6           16.0           10.4           15.0
524288              1              24.594299      21.3           7.0            7.0            10.8           15.0           10.5           14.0
1048576             1              49.639065      21.1           6.0            6.0            10.6           16.0           10.5           13.0
2097152             1              90.493545      23.2           6.0            6.0            12.0           40.0           11.0           13.0
4194304             1              184.397175     22.7           6.0            6.0            11.5           38.0           11.1           19.0
8388608             1              367.256639     22.8           6.0            6.0            10.8           39.0           11.9           22.0
16777216            1              681.912650     24.6           5.0            5.0            11.4           40.0           13.1           17.0
33554432            1              1205.204798    27.8           6.0            6.0            11.3           40.0           16.3           18.0
67108864            1              1636.801561    41.0           5.0            5.0            11.7           36.0           29.2           38.0
[1757492049.615857] [ptyche0321:627915:0]   |                      8177..inf | multi-frag copy-in                                    | tcp/enP6p3s0f1np1 |
[1757492049.615862] [ptyche0321:627915:0]   +--------------------------------+-------------------------------------------------------+-------------------+
"""  # noqa: E501
    (tmp_path / "stdout.txt").write_text(sample)
    df = extract_nixlbench_data(tmp_path / "stdout.txt")
    assert df.shape == (15, 4)


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
