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

import logging

import pandas as pd

from cloudai import System, TestRun

from .predictor.factory import NcclTestPredictorFactory
from .report_generation_strategy import NcclTestReportGenerationStrategy


class NcclTestPredictionReportGenerationStrategy(NcclTestReportGenerationStrategy):
    """Strategy for generating prediction reports from NCCL test outputs."""

    def __init__(self, system: System, tr: TestRun) -> None:
        super().__init__(system, tr)
        self.collective_type = self._normalize_collective_type(tr.test.test_definition.cmd_args.subtest_name)
        self.predictor = NcclTestPredictorFactory.create("default")

    def _normalize_collective_type(self, subtest_name: str) -> str:
        return subtest_name.replace("_perf", "").replace("_mpi", "")

    def generate_report(self) -> None:
        if self.predictor is None:
            logging.warning("Predictor is None. Skipping report generation.")
            return

        df = self._extract_performance_data()
        if df.empty:
            return

        self.predictor.generate(self.test_run, self.collective_type, df)

    def _extract_performance_data(self) -> pd.DataFrame:
        csv_report_path = self.test_run.output_path / "cloudai_nccl_test_csv_report.csv"

        if not csv_report_path.is_file():
            logging.warning(f"Performance CSV {csv_report_path} not found. Prediction report is not generated.")
            return pd.DataFrame()

        df = pd.read_csv(csv_report_path)

        if df.empty:
            logging.warning("CSV load failed. Prediction report is not generated.")

        return df
