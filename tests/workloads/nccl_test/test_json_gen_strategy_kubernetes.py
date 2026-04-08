# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from typing import ClassVar
from unittest.mock import patch

import pytest

from cloudai.core import TestRun
from cloudai.systems.kubernetes import KubernetesSystem
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition, NcclTestKubernetesJsonGenStrategy


class TestNcclTestKubernetesJsonGenStrategy:
    @pytest.fixture
    def test_run(self, k8s_system: KubernetesSystem) -> TestRun:
        cmd_args = NCCLCmdArgs.model_validate({"subtest_name": "all_reduce_perf", "docker_image_url": "fake_image_url"})
        nccl = NCCLTestDefinition(name="name", description="desc", test_template_name="NcclTest", cmd_args=cmd_args)
        tr = TestRun(
            name="t1", test=nccl, nodes=["node1", "node2"], num_nodes=2, output_path=k8s_system.output_path / "t1"
        )
        tr.output_path.mkdir(parents=True, exist_ok=True)
        return tr

    def json_gen_strategy(
        self, kubernetes_system: KubernetesSystem, test_run: TestRun
    ) -> NcclTestKubernetesJsonGenStrategy:
        return NcclTestKubernetesJsonGenStrategy(kubernetes_system, test_run)

    def test_gen_json_basic_structure(self, test_run: TestRun, k8s_system: KubernetesSystem) -> None:
        json_gen_strategy = self.json_gen_strategy(k8s_system, test_run)
        json_payload = json_gen_strategy.gen_json()

        assert json_payload["apiVersion"] == "kubeflow.org/v2beta1"
        assert json_payload["kind"] == "MPIJob"
        assert json_payload["metadata"]["name"] == json_gen_strategy.sanitize_k8s_job_name(test_run.name)
        assert json_payload["spec"]["slotsPerWorker"] == k8s_system.gpus_per_node
        assert json_payload["spec"]["runPolicy"]["cleanPodPolicy"] == "Running"
        assert "Launcher" in json_payload["spec"]["mpiReplicaSpecs"]
        assert "Worker" in json_payload["spec"]["mpiReplicaSpecs"]

    def test_launcher_spec(self, test_run: TestRun, k8s_system: KubernetesSystem) -> None:
        json_payload = self.json_gen_strategy(k8s_system, test_run).gen_json()
        launcher_spec = json_payload["spec"]["mpiReplicaSpecs"]["Launcher"]

        assert launcher_spec["replicas"] == 1

        container = launcher_spec["template"]["spec"]["containers"][0]
        assert container["image"] == "fake_image_url"
        assert container["name"] == "nccl-test-launcher"
        assert container["imagePullPolicy"] == "IfNotPresent"
        assert container["securityContext"]["privileged"] is True
        assert container["command"] == ["/bin/bash", "-c"]

    def test_worker_spec(self, test_run: TestRun, k8s_system: KubernetesSystem) -> None:
        json_gen_strategy = self.json_gen_strategy(k8s_system, test_run)
        json_payload = json_gen_strategy.gen_json()
        worker_spec = json_payload["spec"]["mpiReplicaSpecs"]["Worker"]

        assert worker_spec["replicas"] == test_run.nnodes

        container = worker_spec["template"]["spec"]["containers"][0]
        assert container["image"] == "fake_image_url"
        assert container["name"] == "nccl-test-worker"
        assert container["imagePullPolicy"] == "IfNotPresent"
        assert container["securityContext"]["privileged"] is True
        assert container["command"] == ["/bin/bash", "-c"]
        assert container["args"] == [json_gen_strategy._generate_worker_command()]

        assert container["resources"] == {
            "requests": {"nvidia.com/gpu": str(k8s_system.gpus_per_node)},
            "limits": {"nvidia.com/gpu": str(k8s_system.gpus_per_node)},
        }

        assert container["volumeMounts"] == [{"mountPath": "/dev/shm", "name": "dev-shm"}]
        assert worker_spec["template"]["spec"]["volumes"] == [
            {"name": "dev-shm", "emptyDir": {"medium": "Memory", "sizeLimit": "1Gi"}}
        ]

    def test_env_variables(self, test_run: TestRun, k8s_system: KubernetesSystem) -> None:
        test_run.test.extra_env_vars = {"TEST_VAR": "test_value", "LIST_VAR": ["item1", "item2"]}
        json_payload = self.json_gen_strategy(k8s_system, test_run).gen_json()
        launcher_env = json_payload["spec"]["mpiReplicaSpecs"]["Launcher"]["template"]["spec"]["containers"][0]["env"]
        worker_env = json_payload["spec"]["mpiReplicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]["env"]

        for env_list in [launcher_env, worker_env]:
            env_dict = {item["name"]: item["value"] for item in env_list}
            assert env_dict["OMPI_ALLOW_RUN_AS_ROOT"] == "1"
            assert env_dict["TEST_VAR"] == "test_value"
            assert env_dict["LIST_VAR"] == "item1,item2"

    def test_launcher_command_generation(self, test_run: TestRun, k8s_system: KubernetesSystem) -> None:
        test_run.test.cmd_args = NCCLCmdArgs.model_validate(
            {
                "subtest_name": "all_reduce_perf",
                "docker_image_url": "fake_image_url",
                "nthreads": "4",
                "ngpus": "2",
                "minbytes": "32M",
                "maxbytes": "64M",
            }
        )
        test_run.test.extra_cmd_args = {"extra-flag": "value"}
        json_gen_strategy = self.json_gen_strategy(k8s_system, test_run)
        json_payload = json_gen_strategy.gen_json()
        launcher_args = json_payload["spec"]["mpiReplicaSpecs"]["Launcher"]["template"]["spec"]["containers"][0][
            "args"
        ][0]
        cmd_args = test_run.test.cmd_args

        assert "mpirun" in launcher_args
        assert f"-np {test_run.nnodes * k8s_system.gpus_per_node}" in launcher_args
        assert "-bind-to none" in launcher_args
        assert (
            f"-mca plm_rsh_args '-p {json_gen_strategy.ssh_port}"
            + " -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'"
            in launcher_args
        )
        assert cmd_args.subtest_name in launcher_args
        assert f"--nthreads {cmd_args.nthreads}" in launcher_args
        assert f"--ngpus {cmd_args.ngpus}" in launcher_args


class TestNcclCniNetworking:
    CNI_NETS: ClassVar[list[str]] = ["default/nic0-rail0-plane0", "default/nic0-rail0-plane1"]

    @pytest.fixture
    def test_run(self, k8s_system: KubernetesSystem) -> TestRun:
        cmd_args = NCCLCmdArgs.model_validate({"subtest_name": "all_reduce_perf", "docker_image_url": "img"})
        nccl = NCCLTestDefinition(name="n", description="d", test_template_name="NcclTest", cmd_args=cmd_args)
        tr = TestRun(name="t", test=nccl, nodes=["n1", "n2"], num_nodes=2, output_path=k8s_system.output_path / "t")
        tr.output_path.mkdir(parents=True, exist_ok=True)
        return tr

    def gen_json(self, k8s_system: KubernetesSystem, test_run: TestRun, cni_nets: list[str] | None) -> dict:
        with patch.object(KubernetesSystem, "resolve_cni_networks", return_value=cni_nets):
            return NcclTestKubernetesJsonGenStrategy(k8s_system, test_run).gen_json()

    def test_cni_mode_worker_has_annotations(self, k8s_system: KubernetesSystem, test_run: TestRun) -> None:
        payload = self.gen_json(k8s_system, test_run, self.CNI_NETS)
        worker = payload["spec"]["mpiReplicaSpecs"]["Worker"]
        annotations = worker["template"]["metadata"]["annotations"]
        assert annotations["k8s.v1.cni.cncf.io/networks"] == ",".join(self.CNI_NETS)

    def test_cni_mode_worker_has_nic_resources(self, k8s_system: KubernetesSystem, test_run: TestRun) -> None:
        payload = self.gen_json(k8s_system, test_run, self.CNI_NETS)
        resources = payload["spec"]["mpiReplicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]["resources"]
        for net in self.CNI_NETS:
            nic = net.split("/", 1)[1]
            assert resources["requests"][f"nvidia.com/{nic}"] == "1"
            assert resources["limits"][f"nvidia.com/{nic}"] == "1"

    def test_cni_mode_no_host_network(self, k8s_system: KubernetesSystem, test_run: TestRun) -> None:
        payload = self.gen_json(k8s_system, test_run, self.CNI_NETS)
        worker_spec = payload["spec"]["mpiReplicaSpecs"]["Worker"]["template"]["spec"]
        launcher_spec = payload["spec"]["mpiReplicaSpecs"]["Launcher"]["template"]["spec"]
        assert "hostNetwork" not in worker_spec
        assert "hostNetwork" not in launcher_spec

    def test_host_network_mode_no_annotations(self, k8s_system: KubernetesSystem, test_run: TestRun) -> None:
        payload = self.gen_json(k8s_system, test_run, None)  # resolve_cni_networks returns None → host network
        worker = payload["spec"]["mpiReplicaSpecs"]["Worker"]
        assert "metadata" not in worker["template"]
        assert worker["template"]["spec"]["hostNetwork"] is True

    def test_launcher_never_gets_nic_resources(self, k8s_system: KubernetesSystem, test_run: TestRun) -> None:
        payload = self.gen_json(k8s_system, test_run, self.CNI_NETS)
        launcher_resources = payload["spec"]["mpiReplicaSpecs"]["Launcher"]["template"]["spec"]["containers"][0][
            "resources"
        ]
        assert not any(k.startswith("nvidia.com/nic") for k in launcher_resources["requests"])
