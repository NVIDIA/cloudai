# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Callable, Dict, Type

from .base import BaseDataRepository

_REPOSITORY_REGISTRY: Dict[str, Callable[..., BaseDataRepository]] = {}


def register_repository(name: str):
    def decorator(cls: Type[BaseDataRepository]):
        _REPOSITORY_REGISTRY[name] = cls
        return cls

    return decorator


def get_repository_class(name: str) -> Callable[..., BaseDataRepository]:
    if name not in _REPOSITORY_REGISTRY:
        raise ValueError(f"Data repository backend '{name}' is not registered.")
    return _REPOSITORY_REGISTRY[name]
