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

import os

import pandas as pd
import pytest
from cloudai.schema.test_template.nccl_test.report_generation_strategy import NcclTestReportGenerationStrategy


@pytest.fixture
def setup_test_environment(tmpdir):
    # Create a temporary directory for the test
    test_dir = tmpdir.mkdir("test_env")

    # Create the mock stdout.txt file
    stdout_content = """
    # Using devices
    #  Rank  0 Group  0 Pid 111111 on    server001 device  0 [0xaa] NVIDIA H200 64GB HBM3
    #  Rank  1 Group  0 Pid 222222 on    server001 device  1 [0xbb] NVIDIA H200 64GB HBM3
    #  Rank  2 Group  0 Pid 333333 on    server001 device  2 [0xcc] NVIDIA H200 64GB HBM3
    #  Rank  3 Group  0 Pid 444444 on    server001 device  3 [0xdd] NVIDIA H200 64GB HBM3
    #  Rank  4 Group  0 Pid 555555 on    server001 device  4 [0xee] NVIDIA H200 64GB HBM3
    #  Rank  5 Group  0 Pid 666666 on    server001 device  5 [0xff] NVIDIA H200 64GB HBM3
    #  Rank  6 Group  0 Pid 777777 on    server001 device  6 [0x11] NVIDIA H200 64GB HBM3
    #  Rank  7 Group  0 Pid 888888 on    server001 device  7 [0x22] NVIDIA H200 64GB HBM3
    NCCL version 1.23.4+cuda11.2
    #
    #                                                              out-of-place                       in-place
    #       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
    #        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
         1000000       1000000     float     sum      -1    1.11   10.10   20.20      0    1.12   10.11   20.21      0
         2000000       2000000     float     sum      -1    2.22   20.20   30.30      0    2.23   20.21   30.31      0
         3000000       3000000     float     sum      -1    3.33   30.30   40.40      0    3.34   30.31   40.41      0
         4000000       4000000     float     sum      -1    4.44   40.40   50.50      0    4.45   40.41   50.51      0
         5000000       5000000     float     sum      -1    5.55   50.50   60.60      0    5.56   50.51   60.61      0
         6000000       6000000     float     sum      -1    6.66   60.60   70.70      0    6.67   60.61   70.71      0
         7000000       7000000     float     sum      -1    7.77   70.70   80.80      0    7.78   70.71   80.81      0
         8000000       8000000     float     sum      -1    8.88   80.80   90.90      0    8.89   80.81   90.91      0
         9000000       9000000     float     sum      -1    9.99   90.90  100.10      0   10.00   90.91  100.11      0
        10000000      10000000     float     sum      -1   11.11  100.10  110.20      0   11.12  100.11  110.21      0
        11000000      11000000     float     sum      -1   12.12  110.20  120.30      0   12.13  110.21  120.31      0
        12000000      12000000     float     sum      -1   13.13  120.30  130.40      0   13.14  120.31  130.41      0
    # Out of bounds values : 0 OK
    # Avg bus bandwidth    : 111.111
    #
    """
    stdout_path = os.path.join(test_dir, "stdout.txt")
    with open(stdout_path, "w") as f:
        f.write(stdout_content)

    return test_dir


def test_nccl_report_generation(setup_test_environment):
    test_dir = setup_test_environment

    # Instantiate the strategy
    strategy = NcclTestReportGenerationStrategy()

    # Validate the directory can be handled
    assert strategy.can_handle_directory(test_dir) is True

    # Generate the report
    strategy.generate_report("nccl_test", test_dir)

    # Verify the CSV report
    csv_report_path = os.path.join(test_dir, "cloudai_nccl_test_csv_report.csv")
    assert os.path.isfile(csv_report_path), "CSV report was not generated."

    # Read the CSV and validate the content
    df = pd.read_csv(csv_report_path)
    assert not df.empty, "CSV report is empty."

    # Validate specific values if needed
    # Example: Checking that the first entry matches the expected value
    assert df.iloc[0]["Size (B)"] == 1000000.0, "First row Size (B) does not match."
    assert df.iloc[0]["Algbw (GB/s) Out-of-place"] == 10.10, "First row Algbw (GB/s) Out-of-place does not match."
    assert df.iloc[0]["Busbw (GB/s) Out-of-place"] == 20.20, "First row Busbw (GB/s) Out-of-place does not match."

    # Checking that the last entry matches the expected value
    assert df.iloc[-1]["Size (B)"] == 12000000.0, "Last row Size (B) does not match."
    assert df.iloc[-1]["Algbw (GB/s) Out-of-place"] == 120.30, "Last row Algbw (GB/s) Out-of-place does not match."
    assert df.iloc[-1]["Busbw (GB/s) Out-of-place"] == 130.40, "Last row Busbw (GB/s) Out-of-place does not match."
