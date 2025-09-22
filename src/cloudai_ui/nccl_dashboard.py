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

"""NCCL Dashboard that reuses NcclComparisonReport logic."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from cloudai.core import TestRun, TestScenario
from cloudai.report_generator.comparison_report import ComparisonReportConfig
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nccl_test import NCCLTestDefinition
from cloudai.workloads.nccl_test.nccl_comparison_report import NcclComparisonReport

from .data_layer import TestScenarioInfo


class NCCLDashboard:
    """NCCL Dashboard that reuses existing NcclComparisonReport logic."""

    def __init__(self, scenarios: List[TestScenarioInfo], results_root: Path):
        self.scenarios = scenarios
        self.results_root = results_root

    def get_dashboard_data(self) -> Dict:
        """Get NCCL dashboard data for template rendering."""
        # Collect all NCCL test runs across scenarios
        nccl_test_runs: list[TestRun] = []

        for scenario in self.scenarios:
            for test_run in scenario.test_runs:
                if isinstance(test_run.test.test_definition, NCCLTestDefinition):
                    nccl_test_runs.append(test_run)

        if len(nccl_test_runs) == 0:
            return {
                "test_run_count": 0,
                "gpu_info": None,
                "bokeh_script": "",
                "bokeh_div": "",
            }

        system = SlurmSystem(
            name="slurm", install_path=Path("/"), output_path=Path("/"), default_partition="default", partitions=[]
        )
        test_scenario = TestScenario(name="NCCL", test_runs=nccl_test_runs)

        config = ComparisonReportConfig(enable=True, group_by=["subtest_name"])
        rep = NcclComparisonReport(system, test_scenario, self.results_root, config)
        rep.trs = nccl_test_runs
        bokeh_script, bokeh_div = rep.get_bokeh_html()

        return {
            "test_run_count": len(nccl_test_runs),
            "gpu_info": "TBD",
            "bokeh_script": bokeh_script,
            "bokeh_div": bokeh_div,
        }
