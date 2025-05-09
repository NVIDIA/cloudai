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

from cloudai.registry import Registry

from .kubernetes_installer import KubernetesInstaller
from .kubernetes_job import KubernetesJob
from .kubernetes_runner import KubernetesRunner
from .kubernetes_system import KubernetesSystem

Registry().add_installer("kubernetes", KubernetesInstaller)
Registry().add_system("kubernetes", KubernetesSystem)
Registry().add_runner("kubernetes", KubernetesRunner)

__all__ = [
    "KubernetesJob",
    "KubernetesRunner",
    "KubernetesSystem",
]
