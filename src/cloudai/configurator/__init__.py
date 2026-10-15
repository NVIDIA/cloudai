# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base_agent import BaseAgent as BaseAgent
    from .base_gym import BaseGym as BaseGym
    from .cloudai_gym import CloudAIGymEnv as CloudAIGymEnv
    from .grid_search import GridSearchAgent as GridSearchAgent
    from .gymnasium_adapter import GymnasiumAdapter as GymnasiumAdapter
    from .trajectory import Trajectory as Trajectory

_EXPORTS = {
    "BaseAgent": ".base_agent",
    "BaseGym": ".base_gym",
    "CloudAIGymEnv": ".cloudai_gym",
    "GridSearchAgent": ".grid_search",
    "GymnasiumAdapter": ".gymnasium_adapter",
    "Trajectory": ".trajectory",
}

__all__ = ["BaseAgent", "BaseGym", "CloudAIGymEnv", "GridSearchAgent", "GymnasiumAdapter", "Trajectory"]


def __getattr__(name: str) -> Any:
    """Load convenience exports without pulling execution code into configuration models."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
