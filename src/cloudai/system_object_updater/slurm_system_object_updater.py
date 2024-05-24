# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import cast

from cloudai._core.base_system_object_updater import BaseSystemObjectUpdater
from cloudai._core.system import System
from cloudai.schema.system import SlurmSystem

from .system_object_updater import SystemObjectUpdater


@SystemObjectUpdater.register("slurm")
class SlurmSystemObjectUpdater(BaseSystemObjectUpdater):
    """
    Updater for SLURM scheduler system objects.

    Implements the update method specific to SLURM scheduler systems.
    """

    def update(self, system: System) -> None:
        """
        Update the system object for a SLURM system.

        Args:
            system (System): The system schema object.
        """
        slurm_system = cast(SlurmSystem, system)
        slurm_system.update_node_states()
