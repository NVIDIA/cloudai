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
import re
import socket
from typing import Any, ClassVar, Dict, cast

from cloudai import ReportGenerationStrategy
from cloudai.systems.slurm.slurm_system import SlurmSystem

from .data.http_data_repository import HttpDataRepository
from .data.llama_record_publisher import NeMoRunLLAMARecordPublisher
from .nemo_run import NeMoRunTestDefinition


class NeMoRunDataStoreReportGenerationStrategy(ReportGenerationStrategy):
    """Report generation strategy for NeMoRun LLMB."""

    metrics: ClassVar[list[str]] = ["default", "step-time"]

    def generate_report(self) -> None:
        pass

    def publish_job_data(self) -> None:
        slurm_system = cast(SlurmSystem, self.system)
        if slurm_system.data_repository is None:
            return

        repository_instance = HttpDataRepository(
            slurm_system.data_repository.endpoint,
            slurm_system.data_repository.verify_certs,
        )
        publisher = NeMoRunLLAMARecordPublisher(repository=repository_instance)
        tdef = cast(NeMoRunTestDefinition, self.test_run.test.test_definition)
        docker_image_url = tdef.cmd_args.docker_image_url
        s_fw_version = self.extract_version_from_docker_image(docker_image_url)
        raw_data: Dict[str, Any] = {
            "s_framework": "nemo",
            "s_fw_version": s_fw_version,
            "s_model": tdef.cmd_args.recipe_name,
            "s_model_size": "",  # TODO: 8b, 13b, 30b, 70b...
            "s_workload": tdef.cmd_args.recipe_name,
            "s_dtype": "",  # TODO: fp16, bf16, fp8, fp32
            "s_base_config": "",
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
            "d_metric": "",
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
            "b_synthetic_dataset": "",
        }
        publisher.publish(raw_data)

    def extract_version_from_docker_image(self, docker_image_url: str) -> str:
        version_match = re.search(r":(\d+\.\d+(?:\.\w+)?)", docker_image_url)
        return version_match.group(1) if version_match else "unknown"
