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

from pydantic import BaseModel, ConfigDict

from cloudai import System


class StandaloneSystem(BaseModel, System):
    """
    Class representing a Standalone system.

    This class is used for systems that execute commands directly without a job scheduler.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    install_path: str
    output_path: str
    scheduler: str = "standalone"
    monitor_interval: int = 1

    def update(self) -> None:
        pass
