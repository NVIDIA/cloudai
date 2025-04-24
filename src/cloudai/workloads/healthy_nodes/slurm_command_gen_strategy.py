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

from pathlib import Path
from typing import Dict, List, Union

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy

HEALTHY_NODES_FILE = "cloudai-healthy-nodes.txt"


class HealthyNodesSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for Healthy Nodes on Slurm systems."""

    def _container_mounts(self, tr: TestRun) -> list[str]:
        return []

    def gen_srun_command(self, tr: TestRun) -> str:
        return ""

    def generate_test_command(
        self, env_vars: Dict[str, Union[str, List[str]]], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        return [""]

    def gen_srun_success_check(self, tr: TestRun) -> str:
        # TODO: actual check that there are enough healthy nodes, file shouldn't be created if there are not enough
        # healthy nodes
        healthy_nodes_file = tr.output_path / HEALTHY_NODES_FILE
        check_str = f"[ -f {healthy_nodes_file} ] && echo 1 || echo 0"
        return check_str

    def _set_pre_test_output_path(self, tr: TestRun, base_output_path: Path) -> None:
        tr.output_path = base_output_path

    def pre_test_srun_extra_args(self) -> list[str]:
        # TODO: handle non-containerized tests
        return [f"--nodelist=/cloudai_run_results/{HEALTHY_NODES_FILE}"]
