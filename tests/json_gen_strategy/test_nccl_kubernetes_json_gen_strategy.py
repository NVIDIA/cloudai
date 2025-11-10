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


import pytest

from cloudai.core import TestRun
from cloudai.systems.kubernetes import KubernetesSystem
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition, NcclTestKubernetesJsonGenStrategy


class TestNcclTestKubernetesJsonGenStrategy:
    @pytest.fixture
    def basic_test_run(self, kubernetes_system: KubernetesSystem) -> TestRun:
        cmd_args = NCCLCmdArgs.model_validate({"subtest_name": "all_reduce_perf", "docker_image_url": "fake_image_url"})
        nccl = NCCLTestDefinition(name="name", description="desc", test_template_name="NcclTest", cmd_args=cmd_args)
        return TestRun(name="t1", test=nccl, nodes=["node1", "node2"], num_nodes=2)

    @pytest.fixture
    def test_run_with_env_vars(self, kubernetes_system: KubernetesSystem) -> TestRun:
        cmd_args = NCCLCmdArgs.model_validate({"subtest_name": "all_reduce_perf", "docker_image_url": "fake_image_url"})
        nccl = NCCLTestDefinition(
            name="name",
            description="desc",
            test_template_name="NcclTest",
            cmd_args=cmd_args,
            extra_env_vars={"TEST_VAR": "test_value", "LIST_VAR": ["item1", "item2"]},
        )
        return TestRun(name="t1", test=nccl, nodes=["node1"], num_nodes=1)

    @pytest.fixture
    def test_run_with_extra_args(self, kubernetes_system: KubernetesSystem) -> TestRun:
        cmd_args = NCCLCmdArgs.model_validate(
            {
                "subtest_name": "all_reduce_perf",
                "docker_image_url": "fake_image_url",
                "nthreads": "4",
                "ngpus": "2",
                "minbytes": "32M",
                "maxbytes": "64M",
            }
        )
        nccl = NCCLTestDefinition(
            name="name",
            description="desc",
            test_template_name="NcclTest",
            cmd_args=cmd_args,
            extra_cmd_args={"extra-flag": "value"},
        )
        return TestRun(name="t1", test=nccl, nodes=["node1"], num_nodes=1)

    def json_gen_strategy(
        self, kubernetes_system: KubernetesSystem, test_run: TestRun
    ) -> NcclTestKubernetesJsonGenStrategy:
        return NcclTestKubernetesJsonGenStrategy(kubernetes_system, test_run)

    def test_gen_json_basic_structure(self, basic_test_run: TestRun, kubernetes_system: KubernetesSystem) -> None:
        json_payload = self.json_gen_strategy(kubernetes_system, basic_test_run).gen_json()

        assert json_payload["apiVersion"] == "kubeflow.org/v2beta1"
        assert json_payload["kind"] == "MPIJob"
        assert json_payload["metadata"]["name"] == "nccl-test"
        assert json_payload["spec"]["slotsPerWorker"] == 1
        assert json_payload["spec"]["runPolicy"]["cleanPodPolicy"] == "Running"
        assert "Launcher" in json_payload["spec"]["mpiReplicaSpecs"]
        assert "Worker" in json_payload["spec"]["mpiReplicaSpecs"]

    def test_launcher_spec(self, basic_test_run: TestRun, kubernetes_system: KubernetesSystem) -> None:
        json_payload = self.json_gen_strategy(kubernetes_system, basic_test_run).gen_json()
        launcher_spec = json_payload["spec"]["mpiReplicaSpecs"]["Launcher"]

        assert launcher_spec["replicas"] == 1
        assert launcher_spec["template"]["spec"]["hostNetwork"] is True

        container = launcher_spec["template"]["spec"]["containers"][0]
        assert container["image"] == "fake_image_url"
        assert container["name"] == "nccl-test-launcher"
        assert container["imagePullPolicy"] == "IfNotPresent"
        assert container["securityContext"]["privileged"] is True
        assert container["command"] == ["/bin/bash", "-c"]

    def test_worker_spec(self, basic_test_run: TestRun, kubernetes_system: KubernetesSystem) -> None:
        json_payload = self.json_gen_strategy(kubernetes_system, basic_test_run).gen_json()
        worker_spec = json_payload["spec"]["mpiReplicaSpecs"]["Worker"]

        assert worker_spec["replicas"] == 2
        assert worker_spec["template"]["spec"]["hostNetwork"] is True

        container = worker_spec["template"]["spec"]["containers"][0]
        assert container["image"] == "fake_image_url"
        assert container["name"] == "nccl-test-worker"
        assert container["imagePullPolicy"] == "IfNotPresent"
        assert container["securityContext"]["privileged"] is True
        assert container["ports"][0] == {"containerPort": 2222, "name": "ssh"}
        assert container["command"] == ["/bin/bash"]
        assert container["args"] == ["-c", "/usr/sbin/sshd -p 2222; sleep infinity"]

        assert container["resources"] == {"requests": {"nvidia.com/gpu": "8"}, "limits": {"nvidia.com/gpu": "8"}}

        assert container["volumeMounts"] == [{"mountPath": "/dev/shm", "name": "dev-shm"}]
        assert worker_spec["template"]["spec"]["volumes"] == [
            {"name": "dev-shm", "emptyDir": {"medium": "Memory", "sizeLimit": "1Gi"}}
        ]

    def test_env_variables(self, test_run_with_env_vars: TestRun, kubernetes_system: KubernetesSystem) -> None:
        json_payload = self.json_gen_strategy(kubernetes_system, test_run_with_env_vars).gen_json()
        launcher_env = json_payload["spec"]["mpiReplicaSpecs"]["Launcher"]["template"]["spec"]["containers"][0]["env"]
        worker_env = json_payload["spec"]["mpiReplicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]["env"]

        for env_list in [launcher_env, worker_env]:
            env_dict = {item["name"]: item["value"] for item in env_list}
            assert env_dict["OMPI_ALLOW_RUN_AS_ROOT"] == "1"
            assert env_dict["TEST_VAR"] == "test_value"
            assert env_dict["LIST_VAR"] == "item1,item2"

    def test_launcher_command_generation(
        self, test_run_with_extra_args: TestRun, kubernetes_system: KubernetesSystem
    ) -> None:
        json_payload = self.json_gen_strategy(kubernetes_system, test_run_with_extra_args).gen_json()
        launcher_args = json_payload["spec"]["mpiReplicaSpecs"]["Launcher"]["template"]["spec"]["containers"][0][
            "args"
        ][0]

        assert "mpirun" in launcher_args
        assert "--allow-run-as-root" in launcher_args
        assert "--mca plm_rsh_args '-p 2222'" in launcher_args
        assert "-bind-to none -map-by slot" in launcher_args
        assert "all_reduce_perf" in launcher_args
        assert "--nthreads 4" in launcher_args
        assert "--ngpus 2" in launcher_args
        assert "--minbytes 32M" in launcher_args
        assert "--maxbytes 64M" in launcher_args
        assert "--extra-flag value" in launcher_args
