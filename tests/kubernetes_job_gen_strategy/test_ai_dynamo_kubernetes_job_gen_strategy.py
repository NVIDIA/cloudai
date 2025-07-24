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

from typing import Any, Dict

from cloudai._core.kubernetes_job_gen_strategy import JobSpec
from cloudai.workloads.ai_dynamo import AIDynamoKubernetesJobGenStrategy


def create_test_run(config: Dict[str, Any] | None = None):
    if config is None:
        config = {
            "type": "ai_dynamo",
            "crds_path": "./crds/",
            "platform_path": "./platform/",
            "deployment_path": "./examples/vllm/deploy/agg.yaml",
            "namespace": "dynamo-cloud",
            "docker_server": "nvcr.io",
            "image_tag": "latest",
        }

    class MockTest:
        def __init__(self, config):
            self.config = config
            self.test_definition = self
            self.cmd_args = type("MockCmdArgs", (), config)()
            self.extra_cmd_args = {}
            self.extra_env_vars = {}

    class MockTestRun:
        def __init__(self, name, test):
            self.name = name
            self.test = test
            self.num_nodes = 2

    test = MockTest(config)
    return MockTestRun("test-ai-dynamo", test)


class TestAIDynamoKubernetesJobGenStrategy:
    def test_generate_spec(self):
        strategy = AIDynamoKubernetesJobGenStrategy()
        test_run = create_test_run()
        spec = strategy.generate_spec(test_run)

        assert isinstance(spec, JobSpec)
        assert len(spec.steps) > 1
        assert spec.manifest is None

        # Verify step sequence
        step_names = [step.name for step in spec.steps]
        assert "install_crds" in step_names
        assert "deploy_platform" in step_names
        assert "deploy_vllm" in step_names
        assert "wait_for_pods" in step_names
        assert "port_forward" in step_names
        assert "test_endpoint" in step_names

        # Verify dependencies
        deploy_platform = next(s for s in spec.steps if s.name == "deploy_platform")
        assert "install_crds" in deploy_platform.depends_on

        deploy_vllm = next(s for s in spec.steps if s.name == "deploy_vllm")
        assert "deploy_platform" in deploy_vllm.depends_on

        # Verify helm step configuration
        install_crds = next(s for s in spec.steps if s.name == "install_crds")
        assert install_crds.command_type == "helm"
        assert install_crds.args["action"] == "upgrade"
        assert install_crds.args["release"] == "dynamo-crds"

        # Verify kubectl step configuration
        deploy_vllm = next(s for s in spec.steps if s.name == "deploy_vllm")
        assert deploy_vllm.command_type == "kubectl"
        assert deploy_vllm.args["action"] == "apply"
        assert deploy_vllm.args["namespace"] == "dynamo-cloud"
