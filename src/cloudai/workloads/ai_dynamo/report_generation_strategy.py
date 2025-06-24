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
import shutil

from cloudai.core import ReportGenerationStrategy
from cloudai.systems.slurm.slurm_system import SlurmSystem


class AIDynamoReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from AI Dynamo run directories."""

    def can_handle_directory(self) -> bool:
        output_path = self.test_run.output_path
        csv_files = list(output_path.rglob("profile_genai_perf.csv"))
        json_files = list(output_path.rglob("profile_genai_perf.json"))
        return len(csv_files) > 0 and len(json_files) > 0

    def generate_report(self) -> None:
        output_path = self.test_run.output_path
        source_csv = next(output_path.rglob("profile_genai_perf.csv"))
        target_csv = output_path / "report.csv"

        shutil.copy2(source_csv, target_csv)

        gpus_per_node = None
        if isinstance(self.system, SlurmSystem):
            gpus_per_node = self.system.gpus_per_node

        if gpus_per_node is None:
            logging.warning("gpus_per_node is None, skipping Overall Output Tokens per Second per GPU calculation.")
            return

        num_frontend_nodes = 1
        num_prefill_nodes = self.test_run.test.test_definition.cmd_args.dynamo.prefill_worker.num_nodes
        num_decode_nodes = self.test_run.test.test_definition.cmd_args.dynamo.vllm_worker.num_nodes

        total_gpus = (num_frontend_nodes + num_prefill_nodes + num_decode_nodes) * gpus_per_node

        with open(source_csv, "r") as f:
            lines = f.readlines()
            output_token_throughput_line = next(
                (line for line in lines if "Output Token Throughput (tokens/sec)" in line), None
            )
            if output_token_throughput_line:
                output_token_throughput = float(output_token_throughput_line.split(",")[1].strip())

                overall_output_tokens_per_second_per_gpu = output_token_throughput / total_gpus

                with open(target_csv, "a") as f:
                    f.write(f"Overall Output Tokens per Second per GPU,{overall_output_tokens_per_second_per_gpu}\n")
