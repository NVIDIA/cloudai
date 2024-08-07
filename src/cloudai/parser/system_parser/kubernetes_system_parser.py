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
from typing import Any, Dict

from cloudai import BaseSystemParser
from cloudai.systems.kubernetes import KubernetesSystem


class KubernetesSystemParser(BaseSystemParser):
    """Parser for parsing Kubernetes system configurations."""

    def parse(self, data: Dict[str, Any]) -> KubernetesSystem:
        """
        Parse the Kubernetes system configuration.

        Args:
            data (Dict[str, Any]): The loaded configuration data.

        Returns:
            KubernetesSystem: The parsed Kubernetes system object.

        Raises:
            ValueError: If any mandatory field is missing from the data.
        """

        def get_mandatory_field(field_name: str) -> str:
            value = data.get(field_name)
            if not value:
                raise ValueError(
                    f"Mandatory field '{field_name}' is missing in the Kubernetes system schema. "
                    f"Please ensure that '{field_name}' is present and correctly defined in your system configuration."
                )
            return value

        def safe_int(value):
            try:
                return int(value) if value is not None else None
            except ValueError:
                return None

        def str_to_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            return value.lower() in ("true", "1", "yes")

        # Extract and validate mandatory fields
        name = get_mandatory_field("name")
        install_path = Path(get_mandatory_field("install_path")).resolve()
        output_path = Path(get_mandatory_field("output_path")).resolve()
        default_image = get_mandatory_field("default_image")
        default_namespace = get_mandatory_field("default_namespace")

        # Extract optional fields
        kube_config_path = data.get("kube_config_path")
        if not kube_config_path or len(kube_config_path) == 0 or kube_config_path.isspace():
            home_directory = Path.home()
            kube_config_path = home_directory / ".kube" / "config"
        else:
            kube_config_path = Path(kube_config_path).resolve()

        global_env_vars = data.get("global_env_vars", {})

        return KubernetesSystem(
            name=name,
            install_path=install_path,
            output_path=output_path,
            default_image=default_image,
            kube_config_path=kube_config_path,
            default_namespace=default_namespace,
            global_env_vars=global_env_vars,
        )
