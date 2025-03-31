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
import subprocess
from pathlib import Path

import pandas as pd

from cloudai import TestRun
from cloudai._core.test import PredictorConfig
from cloudai.report_generator.tool.csv_report_tool import CSVReportTool

from .base import NcclTestPredictor


class DefaultNCCLTestPredictor(NcclTestPredictor):
    """Default NCCL test predictor."""

    def generate(self, test_run: TestRun, collective_type: str, df: pd.DataFrame) -> None:
        df = self._extract_performance_data(df)
        if df.empty:
            logging.warning("No valid NCCL performance data extracted. Ensure the test ran successfully.")
            logging.warning("Prediction report not generated.")
            return

        output_path = test_run.output_path
        self._store_intermediate_data(df.drop(columns=["GPU Type", "measured_dur"]), output_path)

        predictions = self.run_predictor(test_run, collective_type, df)
        if predictions.empty:
            logging.warning("Prediction output is empty. Skipping report generation.")
            return

        self._generate_report(df, predictions, output_path)

    def _extract_performance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"GPU Type", "Devices per Node", "Ranks", "Size (B)", "Time (us) Out-of-place"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            logging.warning(f"Missing required columns in performance data: {', '.join(missing_columns)}")
            return pd.DataFrame()
        df.rename(columns={"Size (B)": "message_size", "Time (us) Out-of-place": "measured_dur"}, inplace=True)
        df = df[["GPU Type", "Devices per Node", "Ranks", "message_size", "measured_dur"]]
        return df

    def _store_intermediate_data(self, df: pd.DataFrame, output_path: Path) -> None:
        df = df.rename(columns={"Devices per Node": "num_devices_per_node", "Ranks": "num_ranks"})
        csv_path = output_path / "cloudai_nccl_test_prediction_input.csv"
        df.to_csv(csv_path, index=False)
        logging.debug(f"Stored intermediate predictor input data at {csv_path}")

    def run_predictor(
        self,
        test_run: TestRun,
        collective_type: str,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        test_definition = test_run.test.test_definition
        predictor_config = test_definition.predictor

        if predictor_config is None:
            logging.warning("Predictor configuration is missing. Skipping prediction.")
            return pd.DataFrame()

        venv_path = predictor_config.venv_path
        bin_name = predictor_config.bin_name

        if venv_path is None or bin_name is None:
            logging.warning("Predictor setup is incomplete. Skipping prediction.")
            return pd.DataFrame()

        if "GPU Type" not in df.columns:
            logging.warning("Performance report missing 'GPU Type' column.")
            logging.warning("Prediction report not generated.")
            return pd.DataFrame()
        gpu_type = df["GPU Type"].iloc[0]

        config_path, model_path = self._get_predictor_paths(predictor_config, gpu_type, collective_type)
        if config_path is None or model_path is None:
            logging.warning("Failed to retrieve predictor configuration paths.")
            logging.warning("Prediction report not generated.")
            return pd.DataFrame()

        input_csv = test_run.output_path / "cloudai_nccl_test_prediction_input.csv"
        output_csv = test_run.output_path / "cloudai_nccl_test_prediction_output.csv"
        command = self._build_command(venv_path, bin_name, config_path, model_path, input_csv, output_csv)

        if not self._execute_command(command):
            return pd.DataFrame()

        predictions = self._validate_and_load_predictions(output_csv)

        if input_csv.exists():
            input_csv.unlink()
        if output_csv.exists():
            output_csv.unlink()

        return predictions

    def _build_command(
        self, venv_path: Path, bin_name: str, config_path: Path, model_path: Path, input_csv: Path, output_csv: Path
    ) -> list:
        predictor_path = venv_path / "bin" / bin_name
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
        logging.info(f"Executing predictor command: {' '.join(command)}")
        return command

    def _execute_command(self, command: list) -> bool:
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info(f"Predictor execution successful. Output:\n{result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Predictor execution failed. Error:\n{e.stderr}")
            return False

    def _validate_and_load_predictions(self, output_csv: Path) -> pd.DataFrame:
        if not output_csv.exists():
            logging.error(f"Expected output CSV {output_csv} not found after inference.")
            return pd.DataFrame()

        predictions = pd.read_csv(output_csv)
        required_columns = {"num_devices_per_node", "num_ranks", "message_size", "dur"}
        missing_columns = required_columns - set(predictions.columns)

        if missing_columns:
            logging.error(f"Missing required columns in prediction output: {', '.join(missing_columns)}")
            return pd.DataFrame()

        predictions.rename(columns={"dur": "predicted_dur"}, inplace=True)
        predictions["predicted_dur"] = predictions["predicted_dur"].round(2)

        return predictions[["num_devices_per_node", "num_ranks", "message_size", "predicted_dur"]]

    def _get_predictor_paths(self, predictor_config: PredictorConfig, gpu_type: str, collective_type: str) -> tuple:
        installed_path = predictor_config.git_repo.installed_path
        project_subpath = predictor_config.project_subpath
        if installed_path is None:
            logging.warning("Predictor repository installed path is missing. Skipping path retrieval.")
            return None, None
        if project_subpath is None:
            logging.warning("Predictor repository project subpath is missing. Skipping path retrieval.")
            return None, None
        predictor_sub_path = installed_path / project_subpath
        config_path = predictor_sub_path / f"conf/{gpu_type}/{collective_type}.toml"
        model_path = predictor_sub_path / f"weights/{gpu_type}/{collective_type}.pkl"
        return config_path, model_path

    def _generate_report(self, df: pd.DataFrame, predictions: pd.DataFrame, output_path: Path) -> None:
        df = df.merge(predictions, on="message_size", how="left")
        df["error_ratio"] = ((df["measured_dur"] - df["predicted_dur"]).abs() / df["measured_dur"]).round(2)

        csv_report_tool = CSVReportTool(output_path)
        csv_report_tool.set_dataframe(df[["message_size", "predicted_dur", "measured_dur", "error_ratio"]])
        csv_report_tool.finalize_report(output_path / "cloudai_nccl_test_prediction_csv_report.csv")
