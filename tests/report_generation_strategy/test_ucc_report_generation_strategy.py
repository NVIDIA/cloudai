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

import pytest

from cloudai._core.test import Test
from cloudai._core.test_scenario import TestRun
from cloudai._core.test_template import TestTemplate
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.ucc_test.report_generation_strategy import UCCTestReportGenerationStrategy, parse_ucc_output
from cloudai.workloads.ucc_test.ucc import UCCCmdArgs, UCCTestDefinition

UCC_LOG = """
Collective:             Alltoall
Memory type:            cuda
Datatype:               float32
Reduction:              N/A
Inplace:                0
Warmup:
  small                 100
  large                 20
Iterations:
  small                 1000
  large                 200

       Count        Size                Time, us                           Bandwidth, GB/s
                                 avg         min         max         avg         max         min
           1           4      977.52      952.26     1004.35        0.00        0.00        0.00
           2           8     1002.75      965.62     1041.78        0.00        0.00        0.00
           4          16      976.19      948.68     1006.47        0.00        0.00        0.00
           8          32      969.88      940.82      998.74        0.00        0.00        0.00
          16          64     1001.50      960.72     1031.55        0.00        0.00        0.00
          32         128      998.28      965.55     1027.38        0.01        0.01        0.01
          64         256     1003.58      969.03     1029.33        0.02        0.02        0.02
         128         512     1019.20      985.92     1052.44        0.03        0.03        0.03
         256        1024     1048.66     1014.39     1088.12        0.06        0.06        0.06
         512        2048     1051.25     1011.54     1091.02        0.12        0.13        0.12
        1024        4096     1073.10     1038.11     1111.41        0.24        0.25        0.23
        2048        8192     1136.99     1100.94     1170.66        0.45        0.47        0.44
        4096       16384     1832.97     1751.84     1899.92        0.56        0.59        0.54
        8192       32768     3191.77     3118.83     3292.41        0.65        0.66        0.63
       16384       65536     6206.25     5923.48     6484.98        0.67        0.70        0.64
       32768      131072    12416.39    11509.26    13304.48        0.67        0.72        0.62
       65536      262144    25062.78    23080.53    26787.90        0.66        0.72        0.62
      131072      524288     2218.28     2199.18     2239.80       14.89       15.02       14.75
      262144     1048576     2986.89     2963.94     3013.41       22.12       22.29       21.92
      524288     2097152     4511.98     4485.42     4538.59       29.28       29.46       29.11
     1048576     4194304     7465.54     7439.40     7499.30       35.39       35.52       35.24
     2097152     8388608    13178.06    13137.69    13217.89       40.10       40.23       39.98
     4194304    16777216    24553.88    24515.12    24606.23       43.05       43.11       42.96
     8388608    33554432    47292.49    47258.38    47344.21       44.70       44.73       44.65
"""


@pytest.fixture
def ucc_tr(slurm_system: SlurmSystem) -> TestRun:
    ucc_tr = TestRun(
        name="ucc_test",
        test=Test(
            test_definition=UCCTestDefinition(
                name="ucc_test",
                description="ucc_test",
                test_template_name="ucc_test",
                cmd_args=UCCCmdArgs(),
            ),
            test_template=TestTemplate(system=slurm_system, name="ucc_test"),
        ),
        num_nodes=1,
        nodes=[],
        output_path=slurm_system.output_path,
    )

    ucc_tr.output_path.mkdir(parents=True, exist_ok=True)
    with open(ucc_tr.output_path / "stdout.txt", "w") as f:
        f.write(UCC_LOG)

    return ucc_tr


def test_ucc_report_parsing(slurm_system: SlurmSystem, ucc_tr: TestRun):
    dt = parse_ucc_output(ucc_tr.output_path / "stdout.txt")
    assert dt is not None
    assert len(dt) == 24
    assert dt.iloc[0].tolist() == ["1", "4", "977.52", "952.26", "1004.35", "0.00", "0.00", "0.00"]
    assert dt.iloc[-1].tolist() == [
        "8388608",
        "33554432",
        "47292.49",
        "47258.38",
        "47344.21",
        "44.70",
        "44.73",
        "44.65",
    ]


def test_bokeh_report_generation(slurm_system: SlurmSystem, ucc_tr: TestRun):
    report_gen = UCCTestReportGenerationStrategy(slurm_system, ucc_tr)
    report_gen.generate_report()
    assert (ucc_tr.output_path / "cloudai_ucc_test_bokeh_report.html").exists()
