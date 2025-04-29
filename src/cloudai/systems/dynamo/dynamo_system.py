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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from cloudai._core.base_job import BaseJob
from cloudai._core.system import System
from cloudai.util import CommandShell


class DynamoSystem(BaseModel, System):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    scheduler: str = "dynamo"
    install_path: Path
    output_path: Path
    dynamo_cli: str = "dynamo"
    default_image: Optional[str] = None
    global_env_vars: Dict[str, Any] = {}
    monitor_interval: int = 60
    cmd_shell: CommandShell = Field(default=CommandShell(), exclude=True)

    @field_serializer("install_path", "output_path")
    def _serialize_path(self, v: Path) -> str:
        return str(v)

    def update(self) -> None:
        pass

    def is_job_running(self, job: BaseJob) -> bool:
        cmd = f"{self.dynamo_cli} status {job.id} --format json"
        stdout, _ = self.cmd_shell.execute(cmd).communicate()
        try:
            status_info = json.loads(stdout)
            return status_info.get("state") == "running"
        except Exception:
            return False

    def is_job_completed(self, job: BaseJob) -> bool:
        cmd = f"{self.dynamo_cli} status {job.id} --format json"
        stdout, _ = self.cmd_shell.execute(cmd).communicate()
        try:
            status_info = json.loads(stdout)
            return status_info.get("state") in {"completed", "failed", "cancelled"}
        except Exception:
            return False

    def kill(self, job: BaseJob) -> None:
        cmd = f"{self.dynamo_cli} cancel {job.id}"
        self.cmd_shell.execute(cmd)

    def submit_job(self, job_name: str, spec_path: Path) -> str:
        cmd = f"{self.dynamo_cli} run --spec {spec_path} --name {job_name} --output {self.output_path}"
        stdout, _ = self.cmd_shell.execute(cmd).communicate()
        return stdout.strip()
