# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai._core.kubernetes_job_gen_strategy import JobSpec, JobStep, KubernetesJobGenStrategy
from cloudai._core.test_scenario import TestRun


class AIDynamoKubernetesJobGenStrategy(KubernetesJobGenStrategy):
    """
    Job generation strategy for AI Dynamo workload on Kubernetes systems.

    Handles the multi-step deployment process including CRDs, platform components,
    and VLLM deployment with proper dependencies between steps.
    """

    def generate_spec(self, tr: TestRun) -> JobSpec:
        config = tr.test.config
        return JobSpec(
            steps=[
                JobStep(
                    name="install_crds",
                    command_type="helm",
                    args={
                        "action": "upgrade",
                        "release": "dynamo-crds",
                        "chart_path": config["crds_path"],
                        "namespace": "default",
                        "wait": True,
                    },
                ),
                JobStep(
                    name="deploy_platform",
                    command_type="helm",
                    args={
                        "action": "upgrade",
                        "release": "dynamo-platform",
                        "chart_path": config["platform_path"],
                        "values": {
                            repo_key: (f"{config['docker_server']}/dynamo-operator"),
                            "dynamo-operator.controllerManager.manager.image.tag": config["image_tag"],
                        },
                    },
                    depends_on=["install_crds"],
                ),
                JobStep(
                    name="deploy_vllm",
                    command_type="kubectl",
                    args={"action": "apply", "file": config["deployment_path"], "namespace": config["namespace"]},
                    depends_on=["deploy_platform"],
                ),
                JobStep(
                    name="wait_for_pods",
                    command_type="kubectl",
                    args={
                        "action": "wait",
                        "resource": "pods",
                        "selector": "app=vllm-v1-agg-frontend",
                        "condition": "Ready",
                        "namespace": config["namespace"],
                    },
                    depends_on=["deploy_vllm"],
                ),
                JobStep(
                    name="port_forward",
                    command_type="port_forward",
                    args={
                        "pod_selector": "app=vllm-v1-agg-frontend",
                        "local_port": 8000,
                        "remote_port": 8000,
                        "namespace": config["namespace"],
                    },
                    depends_on=["wait_for_pods"],
                ),
                JobStep(
                    name="test_endpoint",
                    command_type="http",
                    args={
                        "method": "POST",
                        "url": "http://localhost:8000/v1/chat/completions",
                        "headers": {"accept": "application/json", "Content-Type": "application/json"},
                        "body": {"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hello!"}]},
                    },
                    depends_on=["port_forward"],
                ),
            ]
        )


# Define constant at module level
repo_key = "dynamo-operator.controllerManager.manager.image.repository"
