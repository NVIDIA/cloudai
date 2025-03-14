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
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from cloudai import TestDefinition
from cloudai.report_generator.tool.csv_report_tool import CSVReportTool


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

        gpu_type, num_devices, num_ranks = self._extract_device_info()
        df = self._extract_performance_data(gpu_type, num_devices, num_ranks)

        if df.empty:
            logging.warning("No valid NCCL performance data extracted. Ensure the test ran successfully.")
            return

        self._store_intermediate_data(df.drop(columns=["gpu_type", "measured_dur"]))
        predictions = self._run_predictor(gpu_type)

        if predictions.empty:
            logging.warning("Prediction output is empty. Skipping report generation.")
            return

        self._update_performance_report(df, predictions)
        self._generate_prediction_report(df, predictions)

    def _extract_device_info(self) -> Tuple[str, int, int]:
        gpu_type, num_ranks = "Unknown", 0
        device_indices = {}

        if not self.stdout_path.is_file():
            logging.warning(f"stdout file {self.stdout_path} not found. Ensure NCCL test execution before report.")
            return gpu_type, 0, 0

        with self.stdout_path.open(encoding="utf-8") as file:
            for line in file:
                if "Rank" in line and "device" in line and "NVIDIA" in line:
                    num_ranks += 1

                    if match := re.search(r"on\s+([\w\d\-.]+)\s+device\s+(\d+)", line):
                        host, device_index = match.groups()
                        device_indices[host] = max(device_indices.get(host, -1), int(device_index))

                    if match := re.search(r"NVIDIA\s+([A-Z0-9]+)", line):
                        gpu_type = match.group(1).strip()

        num_devices = max(device_indices.values(), default=-1) + 1 if device_indices else 0
        logging.debug(f"Extracted GPU Type: {gpu_type}, Devices per Node: {num_devices}, Ranks: {num_ranks}")
        return gpu_type, num_devices, num_ranks

    def _extract_performance_data(self, gpu_type: str, num_devices: int, num_ranks: int) -> pd.DataFrame:
        if not self.stdout_path.is_file():
            return pd.DataFrame()

        extracted_data = [
            [gpu_type, num_devices, num_ranks, float(match.group(1)), round(float(match.group(2)), 2)]
            for line in self.stdout_path.open(encoding="utf-8")
            if (
                match := re.match(r"^\s*(\d+)\s+\d+\s+\S+\s+\S+\s+[-\d]+\s+\S+\s+\S+\s+\S+\s+\d+\s+(\S+)", line.strip())
            )
        ]

        if not extracted_data:
            logging.debug("No valid NCCL performance data found in stdout.")
            return pd.DataFrame()

        return pd.DataFrame(
            extracted_data, columns=["gpu_type", "num_devices_per_node", "num_ranks", "message_size", "measured_dur"]
        )

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

    def _update_performance_report(self, df: pd.DataFrame, predictions: pd.DataFrame) -> None:
        report_path = self.output_path / "cloudai_nccl_test_csv_report.csv"

        if not report_path.exists():
            logging.warning(f"Performance report {report_path} not found. Skipping update.")
            return

        existing_report = pd.read_csv(report_path)

        if "Size (B)" not in existing_report.columns:
            logging.warning("Missing 'Size (B)' column in existing report. Skipping update.")
            return

        predictions["message_size"] = predictions["message_size"].astype(int)
        df["message_size"] = df["message_size"].astype(int)

        df = df.merge(predictions, on="message_size", how="left", suffixes=("", "_p"))
        df["error_ratio"] = ((df["measured_dur"] - df["predicted_dur"]).abs() / df["measured_dur"]).round(2)

        size_to_metrics = df.set_index("message_size")[["num_devices_per_node", "num_ranks", "gpu_type"]].to_dict(
            orient="index"
        )

        for col in ["num_devices_per_node", "num_ranks", "gpu_type"]:
            if col not in existing_report.columns:
                existing_report[col] = None

        def update_row(row):
            size = row["Size (B)"]
            if size in size_to_metrics:
                row["num_devices_per_node"] = size_to_metrics[size]["num_devices_per_node"]
                row["num_ranks"] = size_to_metrics[size]["num_ranks"]
                row["gpu_type"] = size_to_metrics[size]["gpu_type"]
            return row

        updated_report = existing_report.apply(update_row, axis=1, result_type="expand")
        updated_report.to_csv(report_path, index=False)
        logging.debug(f"Updated performance report saved to {report_path}")

    def _generate_prediction_report(self, df: pd.DataFrame, predictions: pd.DataFrame) -> None:
        df = df.merge(predictions, on="message_size", how="left")
        df["error_ratio"] = ((df["measured_dur"] - df["predicted_dur"]).abs() / df["measured_dur"]).round(2)

        csv_report_tool = CSVReportTool(self.output_path)
        csv_report_tool.set_dataframe(df[["message_size", "predicted_dur", "measured_dur", "error_ratio"]])
        csv_report_tool.finalize_report(Path("cloudai_nccl_test_prediction_csv_report.csv"))

        logging.debug("Saved predictor-based performance prediction report to CSV.")
