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

from typing import Any, Dict, cast

import yaml

from cloudai.core import JsonGenStrategy, TestRun

from .ai_dynamo import AIDynamoTestDefinition


class AIDynamoKubernetesJsonGenStrategy(JsonGenStrategy):
    """JSON generation strategy for AI Dynamo on Kubernetes systems."""

    def gen_json(self, tr: TestRun) -> Dict[Any, Any]:
        td = cast(AIDynamoTestDefinition, tr.test.test_definition)

        if td.cmd_args.dynamo_graph_path is None:
            raise ValueError("dynamo_graph_path must be provided in cmd_args")

        with open(td.cmd_args.dynamo_graph_path, "r") as f:
            yaml_data = yaml.safe_load(f)
            if not isinstance(yaml_data, dict):
                raise ValueError(f"YAML content must be a dictionary/object, got {type(yaml_data)}")
            return yaml_data
