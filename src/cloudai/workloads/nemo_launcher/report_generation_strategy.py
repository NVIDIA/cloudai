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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from cloudai import ReportGenerationStrategy
from cloudai.report_generator.tool.bokeh_report_tool import BokehReportTool


class LLMBReportGenerator:
    """Generate an LLMB report based on training step timings and configuration."""

    SECONDS_PER_DAY = 86400

    def __init__(self, output_path: Path, config: Dict[str, Any], log_path: Path):
        self.output_path = output_path
        self.config = config
        self.log_path = log_path

    def generate(self, train_step_timings: List[float]) -> None:
        conf = self._get_required_configs()
        if conf is None:
            return

        num_nodes = int(conf["num_nodes"])
        num_devices = int(conf["num_devices"])
        dtype = str(conf["dtype"])
        seq_length = int(conf["seq_length"])
        global_batch_size = int(conf["global_batch_size"])
        max_steps = int(conf["max_steps"])
        encoder_seq_length = int(conf["encoder_seq_length"])
        total_gpus = num_nodes * num_devices
        total_tokens = global_batch_size * max_steps * encoder_seq_length

        model, model_size = self._extract_model_info()
        if model is None or model_size is None:
            logging.warning("Skipping LLMB report due to missing model information.")
            return

        model_flops = self._get_model_flops(model, model_size)
        peak_gpu_flops = self._get_peak_gpu_flops(dtype)

        if model_flops is None or peak_gpu_flops is None:
            missing = []
            if model_flops is None:
                missing.append("model_flops")
            if peak_gpu_flops is None:
                missing.append("peak_gpu_flops")
            logging.warning(
                f"Skipping LLMB report due to missing model or hardware specifications: {', '.join(missing)}"
            )
            return

        sft_scheme = "non-lora"  # TODO: adjust if LoRA scheme is used
        throughput = self._compute_throughput(seq_length, global_batch_size, train_step_timings)
        time_to_train = self._compute_time_to_train(total_tokens, throughput)

        mfu = None
        if throughput is not None and throughput > 0:
            mfu = self._compute_mfu(
                global_batch_size,
                float(model_flops),
                total_gpus,
                float(peak_gpu_flops),
                throughput,
                seq_length,
                sft_scheme,
            )

        stats = self._compute_training_step_statistics(train_step_timings)

        model_flops_per_gpu = None
        if mfu is not None:
            model_flops_per_gpu = self._compute_model_flops_per_gpu(float(model_flops), total_gpus)

        self._save_report(
            model,
            model_size,
            dtype,
            num_nodes,
            total_gpus,
            throughput,
            time_to_train,
            mfu,
            model_flops_per_gpu,
            stats,
        )

    def _get_required_configs(self) -> Optional[Dict[str, Any]]:
        keys = [
            (["trainer", "num_nodes"], "num_nodes"),
            (["trainer", "devices"], "num_devices"),
            (["trainer", "precision"], "dtype"),
            (["model", "data", "seq_length"], "seq_length"),
            (["model", "global_batch_size"], "global_batch_size"),
            (["trainer", "max_steps"], "max_steps"),
            (["model", "encoder_seq_length"], "encoder_seq_length"),
        ]
        conf = {}
        missing = []
        for key_list, name in keys:
            val = self._get_config_value(key_list)
            if val is None:
                missing.append(name)
            else:
                conf[name] = val
        if missing:
            logging.warning(f"Skipping LLMB report due to missing configuration values: {', '.join(missing)}")
            return None
        return conf

    def _get_config_value(self, keys: List[str]) -> Optional[Any]:
        sec: Any = self.config
        for key in keys:
            if isinstance(sec, dict) and key in sec:
                sec = sec[key]
            else:
                return None
        return sec

    def _extract_model_info(self) -> Tuple[Optional[str], Optional[str]]:
        if not self.log_path.exists():
            return None, None
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    if "training=" in line:
                        parts = line.split("training=")[-1].strip().split("/")[-1].split("_")
                        model = parts[0] if parts else None
                        model_size = parts[1].replace(" \\", "") if len(parts) > 1 else None
                        return model, model_size
        except Exception as e:
            logging.warning(f"Error while extracting model info: {e}")
            return None, None
        return None, None

    def _get_model_flops(self, model: Optional[str], model_size: Optional[str]) -> Optional[float]:
        model_flops_map = {("llama3", "70b"): 1.70e15}

        if model is None or model_size is None:
            logging.warning("Model or model size is missing. Unable to determine model FLOPS.")
            return None

        model_key = (model.lower(), model_size.lower())

        if model_key not in model_flops_map:
            logging.warning(
                f"Unrecognized model '{model}' with size '{model_size}'. "
                f"Supported models: {list(model_flops_map.keys())}."
            )
            return None

        return model_flops_map[model_key]

    def _get_peak_gpu_flops(self, dtype: Optional[str]) -> Optional[float]:
        m = {"bf16": 9.89e14, "bf16-mixed": 9.89e14, "fp8": 1.979e15}

        if dtype is None:
            logging.warning("Dtype is missing. Unable to determine peak GPU FLOPS.")
            return None

        dtype_lower = dtype.lower()

        if dtype_lower not in m:
            logging.warning(f"Unrecognized dtype '{dtype}'. Supported dtypes: {list(m.keys())}.")
            return None

        return m[dtype_lower]

    def _compute_throughput(self, seq_length: int, global_batch_size: int, timings: List[float]) -> Optional[float]:
        if not timings:
            return None
        avg_train_step_time = float(np.mean(timings))
        if avg_train_step_time == 0:
            return None
        return seq_length * global_batch_size / avg_train_step_time

    def _compute_time_to_train(self, total_tokens: int, throughput: Optional[float]) -> Optional[float]:
        return total_tokens / throughput / self.SECONDS_PER_DAY if throughput else None

    def _compute_mfu(
        self,
        global_batch_size: int,
        model_flops: float,
        total_gpus: int,
        peak_gpu_flops: float,
        throughput: float,
        seq_length: int,
        sft_scheme: str,
    ) -> float:
        mfu_val = (model_flops * throughput) / (seq_length * total_gpus * peak_gpu_flops)
        if sft_scheme == "lora":
            mfu_val *= 2 / 3
        return mfu_val * 100

    def _compute_training_step_statistics(self, timings: List[float]) -> Dict[str, Optional[float]]:
        if not timings:
            return {"Training Step Time Avg": None, "Training Step Time Stdev": None}
        return {
            "Training Step Time Avg": float(np.mean(timings)),
            "Training Step Time Stdev": float(np.std(timings)),
        }

    def _compute_model_flops_per_gpu(self, model_flops: float, total_gpus: int) -> Optional[float]:
        if total_gpus == 0:
            logging.warning("Total GPUs is zero, cannot compute Model FLOPs per GPU.")
            return None
        return model_flops / total_gpus

    @staticmethod
    def _round_if_not_none(value: Optional[float]) -> Optional[float]:
        return round(value, 2) if value is not None else None

    def _save_report(
        self,
        model: str,
        model_size: str,
        dtype: str,
        num_nodes: int,
        total_gpus: int,
        throughput: Optional[float],
        time_to_train: Optional[float],
        mfu: Optional[float],
        model_flops_per_gpu: Optional[float],
        stats: Dict[str, Optional[float]],
    ) -> None:
        data = {
            "Model": model,
            "Model Size": model_size,
            "Dtype": dtype,
            "Nodes": num_nodes,
            "Total GPUs": total_gpus,
            "Throughput (tokens/s)": self._round_if_not_none(throughput),
            "Time to Train (days)": self._round_if_not_none(time_to_train),
            "MFU (%)": self._round_if_not_none(mfu),
            "Model FLOPs per GPU": self._round_if_not_none(model_flops_per_gpu),
        }
        for key, value in stats.items():
            data[key] = self._round_if_not_none(value)

        df = pd.DataFrame([data])
        report_file = self.output_path / "llmb_report.csv"
        df.to_csv(report_file, index=False)
        logging.info(f"LLMB report saved to {report_file}")


class NeMoLauncherReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from NeMo launcher directories."""

    def can_handle_directory(self) -> bool:
        run_dir = self.test_run.output_path / "run"
        return (
            run_dir.exists()
            and run_dir.is_dir()
            and any(file.name.startswith("log") and file.name.endswith(".out") for file in run_dir.iterdir())
        )

    def extract_train_step_timings(self) -> List[float]:
        run_dir = self.test_run.output_path / "run"
        log_file = next((file for file in run_dir.iterdir() if re.match(r"log-.*\.out", file.name)), None)

        if not log_file:
            logging.error("No valid log file found in the run directory")
            return []

        with open(log_file, "r") as f:
            for line in f:
                match = re.search(r"train_step_timing in s: \[([\d.,\s]+)\]", line)
                if match:
                    try:
                        step_timings = [float(val) for val in match.group(1).split(",")]
                        return self._filter_step_timings(step_timings)
                    except ValueError:
                        logging.error(f"Error parsing train step timings in {log_file}")
                        return []

        logging.error(f"No train step timings found in {log_file}")
        return []

    def _filter_step_timings(self, step_timings: List[float]) -> List[float]:
        return step_timings[-20:] if len(step_timings) > 100 else step_timings

    def generate_statistics_report(self, train_step_timings: List[float]) -> None:
        if not train_step_timings:
            return

        stats = {
            "avg": np.mean(train_step_timings),
            "median": np.median(train_step_timings),
            "min": np.min(train_step_timings),
            "max": np.max(train_step_timings),
        }

        summary_file = self.test_run.output_path / "train_step_timing_report.txt"
        with open(summary_file, "w") as f:
            f.writelines([f"{key.capitalize()}: {value:.4f}\n" for key, value in stats.items()])

    def generate_bokeh_report(self, train_step_timings: List[float]) -> None:
        if not train_step_timings:
            return

        df = pd.DataFrame({"Step": range(1, len(train_step_timings) + 1), "train_step_timing in s": train_step_timings})

        report_tool = BokehReportTool(self.test_run.output_path)
        report_tool.add_linear_xy_line_plot(
            title="Train Step Timing over Steps",
            x_column="Step",
            y_column="train_step_timing in s",
            x_axis_label="Step",
            df=df,
            sol=self.test_run.sol,
            color="black",
        )
        report_tool.finalize_report(Path("cloudai_nemo_launcher_bokeh_report.html"))

    def generate_llmb_report(self, train_step_timings: List[float]) -> None:
        yaml_path = self.test_run.output_path / "run" / "run_hydra.yaml"
        log_path = self.test_run.output_path / "run" / "launcher_cmd.log"

        config = self._load_yaml_config(yaml_path)

        if not config:
            logging.error("YAML configuration not found, skipping LLM Benchmark Report.")
            return

        report_generator = LLMBReportGenerator(
            output_path=self.test_run.output_path,
            config=config,
            log_path=log_path,
        )
        report_generator.generate(train_step_timings)

    def _load_yaml_config(self, yaml_path: Path) -> Optional[Dict[str, Any]]:
        if not yaml_path.exists():
            logging.error(f"YAML file not found: {yaml_path}")
            return None

        try:
            with open(yaml_path, "r") as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file: {e}")
            return None

    def generate_report(self) -> None:
        train_step_timings = self.extract_train_step_timings()
        self.generate_statistics_report(train_step_timings)
        self.generate_bokeh_report(train_step_timings)
        self.generate_llmb_report(train_step_timings)
