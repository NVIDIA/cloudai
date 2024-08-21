# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional

from .output_reader_mixin import NcclTestOutputReaderMixin


class KubernetesNcclTestOutputReaderMixin(NcclTestOutputReaderMixin):
    """Mixin to provide output reading functionality for Kubernetes NCCL tests."""

    def _get_stdout_content(self, directory_path: Path) -> Optional[str]:
        aggregated_content = []
        for pod_log_file in directory_path.glob("*-launcher-*.txt"):
            with pod_log_file.open("r") as pod_file:
                aggregated_content.append(pod_file.read())
        return "\n".join(aggregated_content) if aggregated_content else None
