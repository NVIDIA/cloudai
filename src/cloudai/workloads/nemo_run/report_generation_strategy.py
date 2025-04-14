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
from pathlib import Path
from typing import ClassVar, Dict, List, Tuple, cast

import numpy as np

from cloudai import ReportGenerationStrategy
from cloudai._core.test_scenario import METRIC_ERROR
from cloudai.systems.slurm.slurm_system import SlurmSystem

from .nemo_run import NeMoRunTestDefinition


class NeMoRunReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from NeMoRun directories."""

    metrics: ClassVar[List[str]] = ["default", "step-time"]

    @property
    def results_file(self) -> Path:
        return self.test_run.output_path / "stdout.txt"

    def can_handle_directory(self) -> bool:
        for _, __, files in os.walk(self.test_run.output_path):
            for file in files:
                if file.startswith("stdout.txt") and self._parse_timings(self.test_run.output_path / file):
                    return True
        return False

    def generate_report(self) -> None:
        if not self.results_file.exists():
            logging.error(f"{self.results_file} not found")
            return
        data: Dict[str, object] = self._collect_raw_data()
        if not data:
            logging.error(f"No valid step timings found in {self.results_file}. Report generation aborted.")
            return
        self._write_summary_file(cast(Dict[str, float], data["stats"]))
        self._dump_json(data)

    def get_metric(self, metric: str) -> float:
        timings: List[float] = self._parse_timings(self.results_file)
        if not timings or metric not in {"default", "step-time"}:
            return METRIC_ERROR
        return float(np.mean(timings))

    def _collect_raw_data(self) -> Dict[str, object]:
        timings: List[float] = self._parse_timings(self.results_file)
        if not timings:
            return {}
        stats: Dict[str, float] = self._compute_statistics(timings)
        tdef = cast(NeMoRunTestDefinition, self.test_run.test.test_definition)
        slurm_system = cast(SlurmSystem, self.system)
        docker_image_url: str = tdef.cmd_args.docker_image_url
        s_model, s_model_size = self.extract_model_info(tdef.cmd_args.recipe_name)
        data: Dict[str, object] = {
            "s_framework": "nemo",
            "s_fw_version": self.extract_version_from_docker_image(docker_image_url),
            "s_model": s_model,
            "s_model_size": s_model_size,
            "s_workload": tdef.cmd_args.recipe_name,
            "s_dtype": "",  # TODO: fp16, bf16, fp8, fp32
            "s_base_config": "",  # TODO: model.tokenizer.type=/dataset/llama
            "l_max_steps": tdef.cmd_args.trainer.max_steps,
            "l_seq_len": "",  # TODO: should be read from cmd_args
            "l_num_layers": tdef.cmd_args.num_layers,
            "l_vocab_size": "",  # TODO: ./src/cloudperf_resparse/models/nemo/patterns.py
            "l_hidden_size": "",  # TODO: ./src/cloudperf_resparse/models/nemo/patterns.py
            "l_count": "",
            "l_gbs": tdef.cmd_args.data.global_batch_size,
            "l_mbs": tdef.cmd_args.data.micro_batch_size,
            "l_pp": tdef.cmd_args.trainer.strategy.pipeline_model_parallel_size,
            "l_tp": tdef.cmd_args.trainer.strategy.tensor_model_parallel_size,
            "l_vp": tdef.cmd_args.trainer.strategy.virtual_pipeline_model_parallel_size,
            "l_cp": tdef.cmd_args.trainer.strategy.context_parallel_size,
            "d_metric": "",  # TODO: ctx.results.throughput.mean
            "d_metric_stddev": "",
            "d_step_time_mean": "",
            "d_tokens_per_sec": "",  # TODO: = (global_batch_size*encoder_seq_length/throughput.mean)
            "l_checkpoint_size": None,  # TODO: ./common/nemo/nemo-utils.sh
            "d_checkpoint_save_rank_time": None,  # TODO: ./common/nemo/nemo-utils.sh
            "s_job_id": "0",  # TODO: load from metadata when ready
            "s_job_mode": "training",
            "s_image": docker_image_url,
            "l_num_nodes": self.test_run.num_nodes,
            "l_num_gpus": self.test_run.num_nodes * (slurm_system.gpus_per_node or 0),
            "s_cluster": socket.gethostname(),
            "s_user": getpass.getuser(),
            "s_gsw_version": "25.02",
            "b_synthetic_dataset": "true",
            "train_step_timings": timings,
            "stats": stats,
        }
        return data

    def _parse_timings(self, filepath: Path) -> List[float]:
        if not filepath.exists():
            logging.debug(f"{filepath} not found")
            return []
        timings: List[float] = []
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "train_step_timing in s:" in line:
                    try:
                        timing = float(line.split("train_step_timing in s:")[1].strip().split()[0])
                        timings.append(timing)
                    except (ValueError, IndexError):
                        continue
        if not timings:
            logging.debug(f"No train_step_timing found in {filepath}")
            return []
        return self._filter_step_timings(timings)

    def _filter_step_timings(self, step_timings: List[float]) -> List[float]:
        return step_timings[-20:] if len(step_timings) > 100 else step_timings

    def _compute_statistics(self, step_timings: List[float]) -> Dict[str, float]:
        return {
            "avg": float(np.mean(step_timings)),
            "median": float(np.median(step_timings)),
            "min": float(np.min(step_timings)),
            "max": float(np.max(step_timings)),
        }

    def _write_summary_file(self, stats: Dict[str, float]) -> None:
        summary_file: Path = self.test_run.output_path / "report.txt"
        with open(summary_file, "w") as f:
            f.write("Average: {avg}\n".format(avg=stats["avg"]))
            f.write("Median: {median}\n".format(median=stats["median"]))
            f.write("Min: {min}\n".format(min=stats["min"]))
            f.write("Max: {max}\n".format(max=stats["max"]))

    def extract_version_from_docker_image(self, docker_image_url: str) -> str:
        version_match = re.search(r":(\d+\.\d+(?:\.\w+)?)", docker_image_url)
        if version_match:
            return version_match.group(1)
        version_match = re.search(r"__(\d+\.\d+\.\d+)", docker_image_url)
        if version_match:
            return version_match.group(1)
        return "unknown"

    def extract_model_info(self, recipe_name: str) -> Tuple[str, str]:
        size_pattern = re.compile(r"^\d+(?:p\d+)?[bBmM]$")
        tokens: List[str] = recipe_name.split("_")
        for idx, token in enumerate(tokens):
            if size_pattern.match(token):
                s_model: str = "_".join(tokens[:idx])
                s_model_size: str = token
                return s_model, s_model_size
        return recipe_name, ""

    def _dump_json(self, data: Dict[str, object]) -> None:
        data_file: Path = self.test_run.output_path / "report_data.json"
        with open(data_file, "w") as f:
            json.dump(data, f, indent=2)
