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

from datetime import datetime, timezone
from typing import Any, Dict, cast

from cloudai.core import JsonGenStrategy
from cloudai.systems.runai.runai_system import RunAISystem

from .nccl import NCCLTestDefinition


class NcclTestRunAIJsonGenStrategy(JsonGenStrategy):
    """JSON generation strategy for NCCL tests on RunAI systems."""

    def gen_json(self) -> Dict[Any, Any]:
        runai_system = cast(RunAISystem, self.system)
        tdef: NCCLTestDefinition = cast(NCCLTestDefinition, self.test_run.test)
        project_id = runai_system.project_id
        cluster_id = runai_system.cluster_id

        postfix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        name = f"nccl-test-{postfix}"

        training_payload = {
            "name": name,
            "useGivenNameAsPrefix": False,
            "projectId": project_id,
            "clusterId": cluster_id,
            "spec": {
                "command": self.test_run.test.cmd_args.subtest_name,
                "args": " ".join(
                    [
                        f"--{arg} {getattr(self.test_run.test.cmd_args, arg)}"
                        for arg in self.test_run.test.cmd_args.model_dump()
                        if arg not in {"docker_image_url", "subtest_name"}
                    ]
                    + ([self.test_run.test.extra_args_str] if self.test_run.test.extra_cmd_args else [])
                ),
                "image": tdef.docker_image.installed_path,
                "compute": {"gpuDevicesRequest": 8},
                "parallelism": self.test_run.num_nodes,
                "completions": self.test_run.num_nodes,
            },
        }

        return training_payload
