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

from cloudai.core import JsonGenStrategy, TestRun
from cloudai.systems.kubernetes import KubernetesYAMLSystem

from .sleep import SleepCmdArgs, SleepTestDefinition


class SleepKubernetesJsonGenStrategy(JsonGenStrategy):
    """JSON generation strategy for Sleep on Kubernetes systems."""

    def gen_json(self, tr: TestRun) -> Dict[Any, Any]:
        tdef: SleepTestDefinition = cast(SleepTestDefinition, tr.test.test_definition)
        tdef_cmd_args: SleepCmdArgs = tdef.cmd_args
        sec = tdef_cmd_args.seconds

        kubernetes_system = cast(KubernetesYAMLSystem, self.system)

        job_spec = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": tr.name, "namespace": kubernetes_system.default_namespace},
            "spec": {
                "ttlSecondsAfterFinished": 0,
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "args": ["sleep " + str(sec)],
                                "command": ["/bin/bash", "-c"],
                                "image": tr.test.test_definition.cmd_args.docker_image_url,
                                "name": "task",
                            }
                        ],
                        "restartPolicy": "Never",
                    }
                },
            },
        }
        return job_spec
