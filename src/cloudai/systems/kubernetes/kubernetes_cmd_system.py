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

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, cast

from .kubernetes_cmd_job import KubernetesCMDJob

if TYPE_CHECKING:
    pass

from pydantic import BaseModel, ConfigDict

from cloudai.core import BaseJob, System
from cloudai.util import CommandShell


class KubernetesCMDSystem(BaseModel, System):
    """System class for managing Kubernetes command-line based jobs using process IDs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    install_path: Path
    output_path: Path
    default_namespace: str = "default"
    scheduler: str = "kubernetes"
    global_env_vars: Dict[str, Any] = {}
    monitor_interval: int = 1
    _cmd_shell: CommandShell = CommandShell()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    def update(self) -> None:
        """Update the system's state."""
        pass

    def is_job_running(self, job: BaseJob) -> bool:
        k8s_job = cast(KubernetesCMDJob, job)
        try:
            result = self._cmd_shell.execute(f"ps -p {k8s_job.id}").communicate()
            stdout = result[0] if isinstance(result[0], str) else result[0].decode()
            return stdout.count("\n") > 1
        except subprocess.CalledProcessError:
            return False

    def is_job_completed(self, job: BaseJob) -> bool:
        k8s_job = cast(KubernetesCMDJob, job)
        try:
            result = self._cmd_shell.execute(f"ps -p {k8s_job.id}").communicate()
            stdout = result[0] if isinstance(result[0], str) else result[0].decode()
            return stdout.count("\n") <= 1
        except subprocess.CalledProcessError:
            return True

    def kill(self, job: BaseJob) -> None:
        k8s_job = cast(KubernetesCMDJob, job)
        try:
            self._cmd_shell.execute(f"kill {k8s_job.id}").wait(timeout=5)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            self._cmd_shell.execute(f"kill -9 {k8s_job.id}")
