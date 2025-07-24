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
from cloudai.workloads.sleep import SleepKubernetesJobGenStrategy


def create_test_run(config: Dict[str, Any] | None = None):
    if config is None:
        config = {"type": "sleep", "docker_image_url": "ubuntu:latest", "seconds": 60}

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

    test = MockTest(config)
    return MockTestRun("test-sleep", test)


class TestSleepKubernetesJobGenStrategy:
    def test_generate_spec(self):
        strategy = SleepKubernetesJobGenStrategy()
        test_run = create_test_run()
        spec = strategy.generate_spec(test_run)

        assert isinstance(spec, JobSpec)
        assert len(spec.steps) == 1
        assert spec.manifest is not None
        assert spec.manifest["kind"] == "Job"

        step = spec.steps[0]
        assert step.name == "submit_job"
        assert step.command_type == "kubectl"
        assert step.args["action"] == "apply"

        manifest = step.args["manifest"]
        assert manifest["apiVersion"] == "batch/v1"
        assert manifest["spec"]["template"]["spec"]["containers"][0]["args"] == ["sleep 60"]
        assert manifest["spec"]["template"]["spec"]["containers"][0]["image"] == "ubuntu:latest"
