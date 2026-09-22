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

import subprocess
from dataclasses import dataclass, field
from typing import Optional

from cloudai.core import BaseJob


@dataclass
class StandaloneJob(BaseJob):
    """A job class for standalone execution."""

    process: Optional[subprocess.Popen] = field(default=None, compare=False)
    """Handle for the launched process, used to poll completion without spawning a checker.

    ``None`` when the job was not launched by this process -- a dry run, or a job
    reconstructed from prior state -- in which case completion falls back to a signal probe.
    """
