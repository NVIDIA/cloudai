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

from typing import Any, Dict, cast

from cloudai.core import JsonGenStrategy
from cloudai.systems.kubernetes import KubernetesSystem

from .sleep import SleepTestDefinition


class SleepKubernetesJsonGenStrategy(JsonGenStrategy):
    """JSON generation strategy for Sleep on Kubernetes systems."""

    def gen_json(self) -> Dict[Any, Any]:
        tdef: SleepTestDefinition = cast(SleepTestDefinition, self.test_run.test)

        kubernetes_system = cast(KubernetesSystem, self.system)

        job_spec = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": self.sanitize_k8s_job_name(self.test_run.name),
                "namespace": kubernetes_system.default_namespace,
            },
            "spec": {
                "ttlSecondsAfterFinished": 0,
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "args": ["sleep " + str(tdef.cmd_args.seconds)],
                                "command": ["/bin/bash", "-c"],
                                "image": tdef.cmd_args.docker_image_url,
                                "name": "task",
                            }
                        ],
                        "restartPolicy": "Never",
                    }
                },
            },
        }
        return job_spec
