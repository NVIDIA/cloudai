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

from typing import List


def inverse_reward(observation: List[float]) -> float:
    if observation and observation[0] != 0:
        return 1.0 / observation[0]
    return 0.0


def negative_reward(observation: List[float]) -> float:
    if observation:
        return -observation[0]
    return 0.0


def identity_reward(observation: List[float]) -> float:
    if observation:
        return observation[0]
    return 0.0
