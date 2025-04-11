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

import getpass
import json
import logging
import os
import re
import socket
from functools import cache
from pathlib import Path
from typing import ClassVar, cast

import numpy as np

from cloudai import ReportGenerationStrategy
from cloudai._core.test_scenario import METRIC_ERROR
from cloudai.systems.slurm.slurm_system import SlurmSystem

from .nemo_run import NeMoRunTestDefinition


@cache
def extract_timings(stdout_file: Path) -> list[float]:
    if not stdout_file.exists():
        logging.debug(f"{stdout_file} not found")
        return []

    train_step_timings: list[float] = []
    step_timings: list[float] = []

    with open(stdout_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "train_step_timing in s:" in line:
                try:
                    timing = float(line.split("train_step_timing in s:")[1].strip().split()[0])
                    train_step_timings.append(timing)
                    if "global_step:" in line:
                        global_step = int(line.split("global_step:")[1].split("|")[0].strip())
                        if 80 <= global_step <= 100:
                            step_timings.append(timing)
                except (ValueError, IndexError):
                    continue

    if not train_step_timings:
        logging.debug(f"No train_step_timing found in {stdout_file}")
        return []

    if len(step_timings) < 20:
        step_timings = train_step_timings[1:]

    return step_timings


class NeMoRunReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from NeMoRun directories."""

    metrics: ClassVar[list[str]] = ["default", "step-time"]

    def can_handle_directory(self) -> bool:
        for _, __, files in os.walk(self.test_run.output_path):
            for file in files:
                if file.startswith("stdout.txt") and extract_timings(self.test_run.output_path / file):
                    return True
        return False

    @property
    def results_file(self) -> Path:
        return self.test_run.output_path / "stdout.txt"

    def generate_report(self) -> None:
        if not self.results_file.exists():
            logging.error(f"{self.results_file} not found")
            return

        step_timings = extract_timings(self.results_file)
        if not step_timings:
            logging.error(f"No valid step timings found in {self.results_file}. Report generation aborted.")
            return
        stats = self._compute_statistics(step_timings)
        self._write_summary_file(stats)
        raw_data = self._collect_raw_data(step_timings, stats)
        self._dump_raw_data(raw_data)

    def _compute_statistics(self, step_timings: list[float]) -> dict:
        return {
            "avg": np.mean(step_timings),
            "median": np.median(step_timings),
            "min": np.min(step_timings),
            "max": np.max(step_timings),
        }

    def _write_summary_file(self, stats: dict) -> None:
        summary_file = self.test_run.output_path / "report.txt"
        with open(summary_file, "w") as f:
            f.write("Average: {avg}\n".format(avg=stats["avg"]))
            f.write("Median: {median}\n".format(median=stats["median"]))
            f.write("Min: {min}\n".format(min=stats["min"]))
            f.write("Max: {max}\n".format(max=stats["max"]))

    def get_metric(self, metric: str) -> float:
        step_timings = extract_timings(self.results_file)
        if not step_timings:
            return METRIC_ERROR

        if metric not in {"default", "step-time"}:
            return METRIC_ERROR

        return float(np.mean(step_timings))

    def extract_version_from_docker_image(self, docker_image_url: str) -> str:
        version_match = re.search(r":(\d+\.\d+(?:\.\w+)?)", docker_image_url)
        return version_match.group(1) if version_match else "unknown"

    def _collect_raw_data(self, step_timings: list[float], stats: dict) -> dict:
        tdef = cast(NeMoRunTestDefinition, self.test_run.test.test_definition)
        slurm_system = cast(SlurmSystem, self.system)
        docker_image_url = tdef.cmd_args.docker_image_url
        s_fw_version = self.extract_version_from_docker_image(docker_image_url)
        return {
            "s_framework": "nemo",
            "s_fw_version": s_fw_version,
            "s_model": tdef.cmd_args.recipe_name,  # TODO: llama3.1, ...
            "s_model_size": "",  # TODO: 8b, 13b, 30b, 70b...
            "s_workload": tdef.cmd_args.recipe_name,
            "s_dtype": "",  # TODO: fp16, bf16, fp8, fp32
            "s_base_config": "",  # TODO: model.tokenizer.type=/dataset/llama
            "l_max_steps": tdef.cmd_args.trainer.max_steps,
            "l_seq_len": "",  # TODO: ./src/cloudperf_resparse/gsw/log_file_regexes.py
            "l_num_layers": tdef.cmd_args.num_layers,
            "l_vocab_size": "",  # TODO: ./src/cloudperf_resparse/models/nemo/patterns.py
            "l_hidden_size": "",  # TOOD: ./src/cloudperf_resparse/models/nemo/patterns.py
            "l_count": "",
            "l_gbs": "",  # TODO: ./src/cloudperf_resparse/gsw/log_file_regexes.py
            "l_mbs": "",  # TODO: ./src/cloudperf_resparse/gsw/log_file_regexes.py
            "l_pp": tdef.cmd_args.trainer.strategy.pipeline_model_parallel_size,
            "l_tp": tdef.cmd_args.trainer.strategy.tensor_model_parallel_size,
            "l_vp": tdef.cmd_args.trainer.strategy.virtual_pipeline_model_parallel_size,
            "l_cp": "",  # TODO: ./src/cloudperf_resparse/gsw/log_file_regexes.py
            "d_metric": "",  # TODO: ctx.results.throughput.mean
            "d_metric_stddev": "",
            "d_step_time_mean": "",
            "d_tokens_per_sec": "",  # TODO: = (global_batch_size*encoder_seq_length/throughput.mean)
            "l_checkpoint_size": None,  # TODO: ./common/nemo/nemo-utils.sh
            "d_checkpoint_save_rank_time": None,  # TODO: ./common/nemo/nemo-utils.sh
            "s_job_id": "0",  # TODO: load from metadata when ready
            "s_job_mode": "training",
            "s_image": tdef.cmd_args.docker_image_url,
            "l_num_nodes": self.test_run.num_nodes,
            "l_num_gpus": self.test_run.num_nodes * (slurm_system.gpus_per_node or 0),
            "s_cluster": socket.gethostname(),
            "s_user": getpass.getuser(),
            "s_gsw_version": "25.02",
            "b_synthetic_dataset": "",  # TODO: true, false
        }

    def _dump_raw_data(self, raw_data: dict) -> None:
        data_file = self.test_run.output_path / "report_data.json"
        with open(data_file, "w") as f:
            json.dump(raw_data, f, indent=2)
