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

import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from ..._core.test import TestDefinition
from ...report_generator.tool.csv_report_tool import CSVReportTool


class NcclTestPredictionReportGenerator:
    """Generate NCCL test predictor reports by extracting and analyzing performance data."""

    def __init__(self, collective_type: str, output_path: Path, test_definition: TestDefinition):
        self.collective_type = collective_type
        self.output_path = output_path
        self.stdout_path = output_path / "stdout.txt"
        self.test_definition = test_definition
        self.predictor = test_definition.predictor

    def generate(self) -> None:
        if not self.predictor:
            logging.warning("Skipping report generation. Predictor is not installed.")
            return

        df = self._extract_performance_data()
        if df.empty:
            logging.warning("No valid NCCL performance data extracted. Ensure the test ran successfully.")
            return

        self._store_intermediate_data(df.drop(columns=["gpu_type", "measured_dur"]))
        predictions = self._run_predictor(df["gpu_type"].iloc[0])

        if predictions.empty:
            logging.warning("Prediction output is empty. Skipping report generation.")
            return

        self._generate_prediction_report(df, predictions)

    def _extract_performance_data(self) -> pd.DataFrame:
        csv_report_path = self.output_path / "cloudai_nccl_test_csv_report.csv"

        if not csv_report_path.is_file():
            logging.warning(f"Performance CSV {csv_report_path} not found.")
            return pd.DataFrame()

        df = pd.read_csv(csv_report_path)

        required_columns = {"gpu_type", "num_devices_per_node", "num_ranks", "Size (B)", "Time (us) Out-of-place"}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            logging.warning(f"Missing required columns in performance CSV: {', '.join(missing_columns)}")
            return pd.DataFrame()

        df.rename(columns={"Size (B)": "message_size", "Time (us) Out-of-place": "measured_dur"}, inplace=True)

        df = df[["gpu_type", "num_devices_per_node", "num_ranks", "message_size", "measured_dur"]]

        return df

    def _store_intermediate_data(self, df: pd.DataFrame) -> None:
        csv_path = self.output_path / "cloudai_nccl_test_prediction_input.csv"
        df.to_csv(csv_path, index=False)
        logging.debug(f"Stored intermediate predictor input data at {csv_path}")

    def _get_predictor_paths(self, gpu_type: str) -> Optional[Tuple[Path, Path, Path, Path, Path]]:
        if not self.predictor or not self.predictor.git_repo or not self.test_definition.predictor:
            logging.warning("Predictor setup is incomplete. Skipping path retrieval.")
            return None

        installed_path = self.predictor.git_repo.installed_path
        project_subpath = self.test_definition.predictor.project_subpath

        if installed_path is None or project_subpath is None:
            logging.warning("Predictor repository path is missing. Skipping path retrieval.")
            return None

        predictor_sub_path = installed_path / project_subpath
        config_path = predictor_sub_path / f"conf/{gpu_type}/{self.collective_type}.toml"
        model_path = predictor_sub_path / f"weights/{gpu_type}/{self.collective_type}.pkl"
        input_csv = self.output_path / "cloudai_nccl_test_prediction_input.csv"
        output_csv = self.output_path / "cloudai_nccl_test_prediction_output.csv"

        return config_path, model_path, input_csv, output_csv, predictor_sub_path

    def _run_predictor(self, gpu_type: str) -> pd.DataFrame:
        if not self.predictor or not self.predictor.venv_path or not self.test_definition.predictor:
            logging.warning("Predictor setup is incomplete. Skipping prediction.")
            return pd.DataFrame()

        predictor_paths = self._get_predictor_paths(gpu_type)
        if predictor_paths is None:
            return pd.DataFrame()

        config_path, model_path, input_csv, output_csv, predictor_sub_path = predictor_paths

        if not self.predictor or not self.predictor.venv_path:
            logging.warning("Predictor virtual environment is not set up. Skipping prediction.")
            return pd.DataFrame()

        if not self.test_definition.predictor or not self.test_definition.predictor.bin_name:
            logging.warning("Predictor binary is not properly configured. Skipping prediction.")
            return pd.DataFrame()

        venv_path = self.predictor.venv_path
        predictor_path = venv_path / "bin" / self.test_definition.predictor.bin_name
        command = [
            str(predictor_path),
            "--config",
            str(config_path),
            "--model",
            str(model_path),
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--log-level",
            "INFO",
        ]

        logging.debug(f"Running predictor with command: {' '.join(command)}")

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.debug(f"Predictor output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.warning(f"Predictor execution failed. Error:\n{e.stderr}")
            return pd.DataFrame()

        if not output_csv.exists():
            logging.warning(f"Expected output CSV {output_csv} not found after inference.")
            return pd.DataFrame()

        predictions = pd.read_csv(output_csv)
        required_columns = {"num_devices_per_node", "num_ranks", "message_size", "dur"}
        missing_columns = required_columns - set(predictions.columns)

        if missing_columns:
            logging.warning(f"Missing required columns in prediction output: {', '.join(missing_columns)}")
            return pd.DataFrame()

        predictions.rename(columns={"dur": "predicted_dur"}, inplace=True)
        predictions["predicted_dur"] = predictions["predicted_dur"].round(2)

        if input_csv.exists():
            input_csv.unlink()
        if output_csv.exists():
            output_csv.unlink()

        return predictions[["num_devices_per_node", "num_ranks", "message_size", "predicted_dur"]]

    def _generate_prediction_report(self, df: pd.DataFrame, predictions: pd.DataFrame) -> None:
        df = df.merge(predictions, on="message_size", how="left")
        df["error_ratio"] = ((df["measured_dur"] - df["predicted_dur"]).abs() / df["measured_dur"]).round(2)

        csv_report_tool = CSVReportTool(self.output_path)
        csv_report_tool.set_dataframe(df[["message_size", "predicted_dur", "measured_dur", "error_ratio"]])
        csv_report_tool.finalize_report(Path("cloudai_nccl_test_prediction_csv_report.csv"))

        logging.debug("Saved predictor-based performance prediction report to CSV.")
