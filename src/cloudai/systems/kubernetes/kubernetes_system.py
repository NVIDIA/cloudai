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

import getpass
import logging
from typing import Any, Dict, List, Optional, Tuple

from cloudai import System

class KubernetesSystem(System):
    """
    Represents a Slurm system, encapsulating the system's configuration.

    Attributes
        name (str): The name of the Slurm system.
        install_path (str): Installation path of CloudAI software.
        output_path (str): Directory path for output files.
        default_image (str): Default image.
        global_env_vars (Optional[Dict[str, Any]]): Dictionary containing additional configuration settings for the
            system.
    """

    def update(self) -> None:
        """
        Update the system object for a SLURM system.

        This method updates the system object by querying the current state of each node using the 'sinfo' and 'squeue'
        commands, and correlating this information to determine the state of each node and the user running jobs on
        each node.
        """
        pass

    def __init__(
        self,
        name: str,
        install_path: str,
        output_path: str,
        default_image: str,
        kube_config_path: str,
        default_namespace: str,
        global_env_vars: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a KubernetesSystem instance.

        Args:
            name (str): Name of the Slurm system.
            install_path (str): The installation path of CloudAI.
            output_path (str): Path to the output directory.
            default_image (str): Default image.
            kube_config_path(str): Kube config file path.
            global_env_vars (Optional[Dict[str, Any]]): Dictionary containing additional configuration settings for
                the system.
        """
        super().__init__(name, "kubernetes", output_path)
        self.install_path = install_path
        self.default_image = default_image
        self.kube_config_path = kube_config_path
        self.default_namespace = default_namespace
        self.global_env_vars = global_env_vars if global_env_vars is not None else {}
        logging.debug(f"{self.__class__.__name__} initialized")

    def __repr__(self) -> str:
        """
        Provide a structured string representation of the system.
        """
        header = f"System Name: {self.name}\nScheduler Type: {self.scheduler}"
        return header
