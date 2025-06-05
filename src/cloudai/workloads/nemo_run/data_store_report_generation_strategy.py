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

import toml

from cloudai.core import ReportGenerationStrategy
from cloudai.systems.slurm import SlurmSystem
from cloudai.util.lazy_imports import lazy

from .http_data_repository import HttpDataRepository
from .nemo_run import NeMoRunTestDefinition
from .report_generation_strategy import extract_timings


class NeMoRunDataStoreReportGenerationStrategy(ReportGenerationStrategy):
    """Report generation strategy for NeMo 2.0 data store."""

    metrics: ClassVar[list[str]] = []

    @property
    def results_file(self) -> Path:
        return self.test_run.output_path / "stdout.txt"

    def can_handle_directory(self) -> bool:
        for _, __, files in os.walk(self.test_run.output_path):
            for file in files:
                if file.startswith("stdout.txt") and extract_timings(self.test_run.output_path / file):
                    return True
        return False

    def generate_report(self) -> None:
        raw_data = self._collect_raw_data()
        self._dump_json(raw_data)
        self._publish(raw_data)

    def _collect_raw_data(self) -> Dict[str, object]:
        step_timings: List[float] = extract_timings(self.results_file)
        if not step_timings:
            return {}
        tdef = cast(NeMoRunTestDefinition, self.test_run.test.test_definition)
        slurm_system = cast(SlurmSystem, self.system)
        docker_image_url: str = tdef.cmd_args.docker_image_url
        s_model, s_model_size = self.extract_model_info(tdef.cmd_args.recipe_name)
        s_base_config = self.extract_base_config_from_sbatch_script(self.test_run.output_path)
        mean_step_time = float(lazy.np.mean(step_timings))
        global_bs = tdef.cmd_args.data.global_batch_size
        seq_len = tdef.cmd_args.data.seq_length

        if isinstance(global_bs, int) and isinstance(seq_len, int) and mean_step_time > 0:
            tokens_per_sec = global_bs * seq_len / mean_step_time
        else:
            tokens_per_sec = None

        gpus_per_node = (
            slurm_system.gpus_per_node
            if slurm_system.gpus_per_node is not None
            else (slurm_system.ntasks_per_node if slurm_system.ntasks_per_node is not None else 8)
        )

        data: Dict[str, object] = {
            "s_framework": "nemo",
            "s_fw_version": self.extract_version_from_docker_image(docker_image_url),
            "s_model": s_model,
            "s_model_size": s_model_size,
            "s_workload": tdef.cmd_args.recipe_name,
            "s_base_config": s_base_config,
            "l_max_steps": tdef.cmd_args.trainer.max_steps,
            "l_seq_len": tdef.cmd_args.data.seq_length,
            "l_num_layers": tdef.cmd_args.num_layers,
            "l_vocab_size": self.extract_vocab_size(self.results_file),
            "l_gbs": tdef.cmd_args.data.global_batch_size,
            "l_mbs": tdef.cmd_args.data.micro_batch_size,
            "l_pp": tdef.cmd_args.trainer.strategy.pipeline_model_parallel_size,
            "l_tp": tdef.cmd_args.trainer.strategy.tensor_model_parallel_size,
            "l_vp": tdef.cmd_args.trainer.strategy.virtual_pipeline_model_parallel_size,
            "l_cp": tdef.cmd_args.trainer.strategy.context_parallel_size,
            "d_metric": mean_step_time,
            "d_metric_stddev": float(lazy.np.std(step_timings)),
            "d_step_time_mean": mean_step_time,
            "d_tokens_per_sec": tokens_per_sec,
            "s_job_id": self.extract_job_id_from_metadata(self.test_run.output_path),
            "s_job_mode": "training",
            "s_image": docker_image_url,
            "l_num_nodes": self.test_run.num_nodes,
            "l_num_gpus": self.test_run.num_nodes * gpus_per_node,
            "s_cluster": socket.gethostname(),
            "s_user": getpass.getuser(),
            "s_gsw_version": "25.04",
            "b_synthetic_dataset": "true",
        }
        return data

    def extract_version_from_docker_image(self, docker_image_url: str) -> str:
        version_match = re.search(r":(\d+\.\d+(?:\.\w+)?)", docker_image_url)
        if version_match:
            return version_match.group(1)
        version_match = re.search(r"__(\d+\.\d+\.\d+)", docker_image_url)
        if version_match:
            return version_match.group(1)
        return "unknown"

    def extract_model_info(self, recipe_name: str) -> Tuple[str, str]:
        recipe_name = recipe_name.removeprefix("cloudai_")
        size_pattern = re.compile(r"^\d+(?:p\d+)?[bBmM]$")
        tokens: List[str] = recipe_name.split("_")
        for idx, token in enumerate(tokens):
            if size_pattern.match(token):
                s_model: str = "_".join(tokens[:idx])
                s_model_size: str = token
                return s_model, s_model_size
        return recipe_name, ""

    def extract_base_config_from_sbatch_script(self, output_path: Path) -> str:
        sbatch_script = output_path / "cloudai_sbatch_script.sh"
        if not sbatch_script.is_file():
            logging.warning(f"SBATCH script not found: {sbatch_script}")
            return ""

        try:
            last_command = self._read_last_nonempty_line(sbatch_script)
            return self._extract_config_from_command_line(last_command)
        except Exception as e:
            logging.exception(f"Error extracting base config from sbatch script: {e}")
            return ""

    def _read_last_nonempty_line(self, file_path: Path) -> str:
        with file_path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines[-1] if lines else ""

    def _extract_config_from_command_line(self, command_line: str) -> str:
        y_flag = " -y"
        index = command_line.find(y_flag)
        if index == -1:
            return ""
        return command_line[index + len(y_flag) :].strip()

    def extract_vocab_size(self, filepath: Path) -> int:
        if not filepath.exists():
            return -1
        vocab_line_pattern = re.compile(r"Padded vocab_size:\s*(\d+)", re.IGNORECASE)

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                match = vocab_line_pattern.search(line)
                if match:
                    return int(match.group(1))
        return -1

    def extract_job_id_from_metadata(self, output_path: Path) -> str:
        metadata_dir = output_path / "metadata"
        if not metadata_dir.is_dir():
            logging.debug(f"No metadata directory found at {metadata_dir}")
            return "0"

        toml_files = sorted(metadata_dir.glob("*.toml"))
        if not toml_files:
            logging.debug(f"No TOML files found in metadata directory: {metadata_dir}")
            return "0"

        metadata_file = toml_files[0]
        try:
            metadata = toml.load(metadata_file)
            job_id = metadata.get("slurm", {}).get("job_id")
            if job_id is not None:
                return str(job_id)
            logging.debug(f"'job_id' not found in 'slurm' section of {metadata_file}")
        except (toml.TomlDecodeError, OSError) as e:
            logging.warning(f"Failed to read or parse {metadata_file}: {e}")

        return "0"

    def _dump_json(self, data: Dict[str, object]) -> None:
        data_file: Path = self.test_run.output_path / "report_data.json"
        with open(data_file, "w") as f:
            json.dump(data, f, indent=2)

    def _publish(self, raw_data: Dict) -> None:
        slurm_system = cast(SlurmSystem, self.system)
        if slurm_system.data_repository is None:
            return

        try:
            repository = HttpDataRepository(
                slurm_system.data_repository.endpoint,
                slurm_system.data_repository.verify_certs,
            )
        except ValueError as e:
            logging.warning("Skipping data publish: %s", e)
            return

        repository.push(raw_data)
