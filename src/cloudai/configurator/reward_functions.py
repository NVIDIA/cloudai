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

import math
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


def ai_dynamo_reward(observation: List[float]) -> float:
    # Metric indices
    TTFT_IDX = 0
    ITL_IDX = 1

    # Baseline expected values (for normalization)
    TTFT_BASELINE = 0.3  # seconds
    ITL_BASELINE = 0.02  # seconds

    # Latency thresholds (for hard penalties)
    TTFT_SLA = 0.8  # seconds
    ITL_SLA = 0.1  # seconds

    # Weighting between TTFT and ITL in the final reward
    TTFT_WEIGHT = 0.4
    ITL_WEIGHT = 0.6

    # Extract metric values and guard against zero or negative values
    ttft = max(observation[TTFT_IDX], 1e-4)
    itl = max(observation[ITL_IDX], 1e-4)

    # Enforce SLA thresholds with hard penalties
    if ttft > TTFT_SLA or itl > ITL_SLA:
        return -10.0

    # Compute normalized log penalties
    ttft_penalty = math.log(ttft / TTFT_BASELINE + 1)
    itl_penalty = math.log(itl / ITL_BASELINE + 1)

    # Weighted combined reward (lower is better, hence negative)
    reward = -(TTFT_WEIGHT * ttft_penalty + ITL_WEIGHT * itl_penalty)

    return reward
