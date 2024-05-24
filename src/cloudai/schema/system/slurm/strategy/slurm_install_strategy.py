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

from typing import Any, Dict, cast

from cloudai import InstallStrategy, System
from cloudai.schema.system import SlurmSystem


class SlurmInstallStrategy(InstallStrategy):
    """
    Abstract base class for defining installation strategies specific to Slurm environments.

    Attributes
        slurm_system (SlurmSystem): A casted version of the `system` attribute,
                                    which provides Slurm-specific properties
                                    and methods.
    """

    def __init__(
        self,
        system: System,
        env_vars: Dict[str, Any],
        cmd_args: Dict[str, Any],
    ) -> None:
        super().__init__(system, env_vars, cmd_args)
        self.slurm_system = cast(SlurmSystem, self.system)
