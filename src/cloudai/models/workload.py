# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cloudai.core import GitRepo, Installable, JobStatusResult, PythonExecutable, TestRun


class AgentConfig(BaseModel):
    """Base configuration class for agents used in DSE."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Common agent parameters
    random_seed: Optional[int] = None
    
    # Allow for additional agent-specific parameters
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class BOAgentConfig(AgentConfig):
    """Configuration for Bayesian Optimization agent."""
    
    # BO-specific parameters
    sobol_num_trials: Optional[int] = None
    botorch_num_trials: Optional[int] = None
    
    # Seed parameters for starting optimization from known configuration
    seed_parameters: Optional[Dict[str, Any]] = None
    
    # Allow for additional agent-specific parameters
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class CmdArgs(BaseModel):
    """Test command arguments."""

    model_config = ConfigDict(extra="allow")


class NsysConfiguration(BaseModel):
    """NSYS configuration."""

    model_config = ConfigDict(extra="forbid")

    enable: bool = True
    nsys_binary: str = "nsys"
    task: str = "profile"
    output: Optional[str] = None
    sample: Optional[str] = None
    trace: Optional[str] = None
    force_overwrite: Optional[bool] = None
    capture_range: Optional[str] = None
    capture_range_end: Optional[str] = None
    cuda_graph_trace: Optional[str] = None
    gpu_metrics_devices: Optional[str] = None
    extra_args: list[str] = []

    @property
    def cmd_args(self) -> list[str]:
        parts = [f"{self.nsys_binary}", f"{self.task}"]
        if self.sample:
            parts.append(f"-s {self.sample}")
        if self.output:
            parts.append(f"-o {self.output}")
        if self.trace:
            parts.append(f"-t {self.trace}")
        if self.force_overwrite is not None:
            parts.append(f"--force-overwrite={str(self.force_overwrite).lower()}")
        if self.capture_range:
            parts.append(f"--capture-range={self.capture_range}")
        if self.capture_range_end:
            parts.append(f"--capture-range-end={self.capture_range_end}")
        if self.cuda_graph_trace:
            parts.append(f"--cuda-graph-trace={self.cuda_graph_trace}")
        if self.gpu_metrics_devices:
            parts.append(f"--gpu-metrics-devices={self.gpu_metrics_devices}")
        parts.extend(self.extra_args)

        return parts


@dataclass
class PredictorConfig(PythonExecutable):
    """Predictor configuration."""

    bin_name: Optional[str] = None

    def __hash__(self) -> int:
        """
        Hash the PredictorConfig.

        It is based on git repo on purpose to avoid re-downloading the same repo for multiple scripts.
        """
        return self.git_repo.__hash__()


class TestDefinition(BaseModel, ABC):
    """Base Test object."""

    __test__ = False

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    test_template_name: str
    cmd_args: Any
    extra_env_vars: dict[str, Union[str, List[str]]] = {}
    extra_cmd_args: dict[str, str] = {}
    extra_container_mounts: list[str] = []
    git_repos: list[GitRepo] = []
    nsys: Optional[NsysConfiguration] = None
    predictor: Optional[PredictorConfig] = None
    agent: str = "grid_search"
    agent_steps: int = 1
    agent_metrics: list[str] = Field(default=["default"])
    agent_reward_function: str = "inverse"
    agent_config: Optional[AgentConfig] = None

    @field_validator('agent_config', mode='before')
    @classmethod
    def parse_agent_config(cls, v, info):
        """Parse agent_config based on the agent type."""
        import logging
        
        if v is None:
            return None
            
        if isinstance(v, AgentConfig):
            return v
            
        if isinstance(v, dict):
            agent_type = info.data.get('agent', 'grid_search')
            
            # Critical debugging: Track when BO data is incomplete
            if agent_type == 'bo_gp':
                has_bo_fields = 'sobol_num_trials' in v or 'botorch_num_trials' in v or 'seed_parameters' in v
                if not has_bo_fields:
                    logging.warning(f"🚨 BO agent_config missing BO fields! Input: {v}")
                else:
                    logging.info(f"✅ BO agent_config has BO fields: {v}")
            
            agent_config_map = {
                'bo_gp': BOAgentConfig
            }
            
            config_class = agent_config_map.get(agent_type, AgentConfig)
            result = config_class.model_validate(v)
            
            return result
            
        return v

    def resolve_seed_parameters(self, action_space: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Resolve seed parameters by extracting values from the action space.
        
        Args:
            action_space: The flattened action space from cmd_args
            
        Returns:
            Resolved seed parameters with actual values
        """
        if not self.agent_config or not hasattr(self.agent_config, 'seed_parameters'):
            return None
            
        seed_params = self.agent_config.seed_parameters
        if not seed_params:
            return None
            
        resolved = {}
        for param_name, value_spec in seed_params.items():
            if param_name in action_space:
                param_options = action_space[param_name]
                if isinstance(param_options, list):
                    if isinstance(value_spec, int) and 0 <= value_spec < len(param_options):
                        resolved[param_name] = param_options[value_spec]
                    elif value_spec in param_options:
                        resolved[param_name] = value_spec
                    else:
                        resolved[param_name] = param_options[0]
                else:
                    resolved[param_name] = param_options
            else:
                resolved[param_name] = value_spec
                
        return resolved

    @property
    def cmd_args_dict(self) -> Dict[str, Union[str, List[str]]]:
        return self.cmd_args.model_dump()

    @property
    def extra_args_str(self) -> str:
        parts = []
        for k, v in self.extra_cmd_args.items():
            parts.append(f"{k}={v}" if v else k)
        return " ".join(parts)

    @property
    def installables(self) -> list[Installable]:
        return [*self.git_repos]

    def constraint_check(self, tr: TestRun) -> bool:
        return True

    @property
    def is_dse_job(self) -> bool:
        def check_dict(d: dict) -> bool:
            if isinstance(d, dict):
                for value in d.values():
                    if isinstance(value, list) or (isinstance(value, dict) and check_dict(value)):
                        return True
            return False

        return check_dict(self.cmd_args_dict) or check_dict(self.extra_env_vars)

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        return JobStatusResult(is_successful=True)
