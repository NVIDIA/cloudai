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
from cloudai.systems.kubernetes import KubernetesNode, KubernetesNodeState, KubernetesSystem


class KubernetesSystemParser(BaseSystemParser):
    """Parser for parsing Kubernetes system configurations."""

    def parse(self, data: Dict[str, Any]) -> KubernetesSystem:  # noqa: C901
        """
        Parse the Kubernetes system configuration.

        Args:
            data (Dict[str, Any]): The loaded configuration data.

        Returns:
            KubernetesSystem: The parsed Kubernetes system object.

        Raises:
            ValueError: If 'name' or 'partitions' are missing from the data or if there are node list parsing issues
            or group membership conflicts.
        """

        return KubernetesSystem(
            name=name,
            install_path=install_path,
            output_path=output_path,
            default_partition=default_partition,
            partitions=updated_partitions,
            account=account,
            distribution=distribution,
            mpi=mpi,
            gpus_per_node=gpus_per_node,
            ntasks_per_node=ntasks_per_node,
            cache_docker_images_locally=cache_docker_images_locally,
            groups=updated_groups,
            global_env_vars=global_env_vars,
        )
