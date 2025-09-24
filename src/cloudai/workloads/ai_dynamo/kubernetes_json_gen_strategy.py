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

from pathlib import Path
from typing import Any, Dict, Protocol, Union, cast, runtime_checkable

import yaml

from cloudai.core import JsonGenStrategy, TestRun

from .ai_dynamo import AIDynamoTestDefinition


@runtime_checkable
class HasDynamoGraphPath(Protocol):
    """Protocol for objects that have a dynamo_graph_path attribute."""

    dynamo_graph_path: Union[str, Path]


class AIDynamoKubernetesJsonGenStrategy(JsonGenStrategy):
    """JSON generation strategy for AI Dynamo on Kubernetes systems."""

    def gen_json(self, tr: TestRun) -> Dict[Any, Any]:
        td = cast(AIDynamoTestDefinition, tr.test.test_definition)

        cmd_args = td.cmd_args
        if not isinstance(cmd_args, HasDynamoGraphPath) or not hasattr(cmd_args, "dynamo_graph_path"):
            raise ValueError("dynamo_graph_path must be provided in cmd_args")

        graph_path = Path(cast(HasDynamoGraphPath, cmd_args).dynamo_graph_path)
        if not graph_path.exists():
            raise ValueError(f"Dynamo graph file not found at {graph_path}")

        with open(graph_path, "r") as f:
            yaml_data = yaml.safe_load(f)
            if not isinstance(yaml_data, dict):
                raise ValueError(f"YAML content must be a dictionary/object, got {type(yaml_data)}")
            return yaml_data
