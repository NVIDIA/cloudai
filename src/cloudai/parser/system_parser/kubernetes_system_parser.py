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

import os
from typing import Any, Dict, List

from cloudai import BaseSystemParser
from cloudai.systems.kubernetes import KubernetesSystem


class KubernetesSystemParser(BaseSystemParser):
    """Parser for parsing Slurm system configurations."""

    def parse(self, data: Dict[str, Any]) -> KubernetesSystem:  # noqa: C901
        """
        Parse the Slurm system configuration.

        Args:
            data (Dict[str, Any]): The loaded configuration data.

        Returns:
            KubernetesSystem: The parsed kubernetes system object.

        Raises:
            ValueError: If 'name' is missing from the data or if there are node list parsing issues
            or group membership conflicts.
        """

        def safe_int(value):
            try:
                return int(value) if value is not None else None
            except ValueError:
                return None

        def str_to_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            return value.lower() in ("true", "1", "yes")

        name = data.get("name")
        if not name:
            raise ValueError("Missing mandatory field: 'name'")

        install_path = data.get("install_path")
        if not install_path:
            raise ValueError("Field 'install_path' is required.")
        install_path = os.path.abspath(install_path)

        output_path = data.get("output_path")
        if not output_path:
            raise ValueError("Field 'output_path' is required.")
        output_path = os.path.abspath(output_path)

        default_image = data.get("default_image")
        if not default_image:
            raise ValueError("Field 'default_image' is required.")

        kube_config_path = data.get("kube_config_path")
        if not kube_config_path or len(kube_config_path) == 0 or kube_config_path.isspace():
            home_directory = os.path.expanduser('~')
            kube_config_path = os.path.join(home_directory, '.kube', 'config')

        default_namespace = data.get("default_namespace")
        if not default_namespace:
            raise ValueError("Field 'default_namespace' is required.")

        global_env_vars = data.get("global_env_vars", {})
        
        return KubernetesSystem(
            name=name,
            install_path=install_path,
            output_path=output_path,
            default_image=default_image,
            kube_config_path=kube_config_path,
            default_namespace=default_namespace,
            global_env_vars = global_env_vars,
        )
