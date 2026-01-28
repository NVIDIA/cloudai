# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from abc import ABC
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class AgentConfig(BaseModel, ABC):
    """Base configuration for agent overrides."""

    model_config = ConfigDict(extra="forbid")
    random_seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class GeneticAlgorithmConfig(AgentConfig):
    """Configuration overrides for Genetic Algorithm agent."""

    population_size: Optional[int] = Field(default=None, ge=2, description="Population size for the genetic algorithm")
    n_offsprings: Optional[int] = Field(default=None, ge=1, description="Number of offsprings per generation")
    crossover_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Crossover probability")
    mutation_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Mutation probability")


class BayesianOptimizationConfig(AgentConfig):
    """Configuration overrides for Bayesian Optimization agent."""

    sobol_num_trials: Optional[int] = Field(default=None, ge=1, description="Number of SOBOL initialization trials")
    botorch_num_trials: Optional[int] = Field(
        default=None, description="Number of BoTorch trials (-1 for unlimited until max_steps)"
    )


class MultiArmedBanditConfig(AgentConfig):
    """Configuration overrides for Multi-Armed Bandit agent."""

    algorithm: Optional[str] = Field(
        default=None,
        description="MAB algorithm: ucb1, ts (thompson_sampling), epsilon_greedy, softmax, or random",
    )
    algorithm_params: Optional[dict[str, Any]] = Field(
        default=None, description="Algorithm-specific parameters (e.g., alpha for UCB1, epsilon for epsilon_greedy)"
    )
    seed_parameters: Optional[dict[str, Any]] = Field(
        default=None, description="Initial seed configuration to evaluate first"
    )
    max_arms: Optional[int] = Field(default=None, ge=1, description="Maximum number of arms in the action space")
    warm_start_size: Optional[int] = Field(
        default=None, ge=0, description="Number of arms to randomly explore initially"
    )
    epsilon_override: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Epsilon value for exploration (overrides algorithm epsilon)"
    )
    max_explore_steps: Optional[int] = Field(
        default=None, ge=0, description="Maximum steps for epsilon exploration (None for unlimited)"
    )
    prefer_unseen_random: Optional[bool] = Field(
        default=None, description="Prefer unseen arms during random exploration (epsilon)"
    )
