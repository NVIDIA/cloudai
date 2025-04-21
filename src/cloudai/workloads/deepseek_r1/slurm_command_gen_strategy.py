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

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy

from .deepseek_r1 import DeepSeekR1TestDefinition


class DeepSeekR1SlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for DeepSeekR1."""

    def _container_mounts(self, tr: TestRun) -> List[str]:
        tdef: DeepSeekR1TestDefinition = cast(DeepSeekR1TestDefinition, tr.test.test_definition)
        model_name = tdef.extra_env_vars.get("NIM_MODEL_NAME")

        if not isinstance(model_name, str) or not model_name.strip():
            return []

        host_path = Path(model_name)
        if not host_path.is_dir():
            raise FileNotFoundError(f"Model directory not found at: {host_path}")

        return [f"{model_name}:{model_name}:ro"]

    def _parse_slurm_args(
        self,
        job_name_prefix: str,
        env_vars: Dict[str, Union[str, List[str]]],
        cmd_args: Dict[str, Union[str, List[str]]],
        tr: TestRun,
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

        tdef: DeepSeekR1TestDefinition = cast(DeepSeekR1TestDefinition, tr.test.test_definition)
        base_args.update({"image_path": tdef.docker_image.installed_path})

        return base_args

    def _append_sbatch_directives(
        self, batch_script_content: List[str], args: Dict[str, Any], output_path: Path
    ) -> None:
        super()._append_sbatch_directives(batch_script_content, args, output_path)

        batch_script_content.append("export HEAD_NODE=$SLURM_JOB_MASTER_NODE")
        batch_script_content.append("export NIM_LEADER_IP_ADDRESS=$SLURM_JOB_MASTER_NODE")
        batch_script_content.append(f"export NIM_NUM_COMPUTE_NODES={args['num_nodes']}")

        ngc_api_key = self._read_ngc_api_key()
        if ngc_api_key:
            batch_script_content.append(f"export NGC_API_KEY={ngc_api_key}")
        else:
            batch_script_content.append("echo 'WARNING: Failed to load NGC API key'")

    def _read_ngc_api_key(self) -> Optional[str]:
        credentials_path = Path.home() / ".config" / "enroot" / ".credentials"
        if not credentials_path.exists():
            return None

        try:
            with credentials_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("machine nvcr.io"):
                        parts = line.strip().split()
                        if "password" in parts:
                            idx = parts.index("password")
                            if idx + 1 < len(parts):
                                return parts[idx + 1]
        except Exception:
            return None

        return None

    def generate_test_command(
        self, env_vars: Dict[str, Union[str, List[str]]], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        return ["/opt/nim/start_server.sh"]
