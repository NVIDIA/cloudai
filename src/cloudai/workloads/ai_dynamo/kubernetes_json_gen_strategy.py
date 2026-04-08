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

from typing import Any, Dict, cast

import yaml

from cloudai.core import JsonGenStrategy
from cloudai.systems.kubernetes import KubernetesSystem

from .ai_dynamo import AIDynamoTestDefinition, WorkerBaseArgs, WorkerConfig


class AIDynamoKubernetesJsonGenStrategy(JsonGenStrategy):
    """JSON generation strategy for AI Dynamo on Kubernetes systems."""

    DEPLOYMENT_FILE_NAME = "deployment.yaml"

    def gen_frontend_dict(self, cni_networks: list[str] | None = None) -> dict[str, Any]:
        system = cast(KubernetesSystem, self.system)
        tdef = cast(AIDynamoTestDefinition, self.test_run.test)
        cfg: dict[str, Any] = {
            "dynamoNamespace": system.default_namespace,
            "componentType": "frontend",
            "replicas": 1,
            "extraPodSpec": {
                "mainContainer": {
                    "image": tdef.cmd_args.docker_image_url,
                }
            },
        }
        if cni_networks is None:
            cfg["extraPodSpec"]["hostNetwork"] = True
        return cfg

    def gen_decode_dict(self, cni_networks: list[str] | None = None) -> dict[str, Any]:
        tdef = cast(AIDynamoTestDefinition, self.test_run.test)

        decode_cfg = self._get_base_service_dict(cni_networks)
        decode_cfg["extraPodSpec"]["mainContainer"]["command"] = tdef.cmd_args.dynamo.decode_worker.cmd.split()

        args = ["--model", tdef.cmd_args.dynamo.model]
        if tdef.cmd_args.dynamo.prefill_worker:
            decode_cfg["subComponentType"] = "decode-worker"
            args.append("--is-decode-worker")
        args.extend(self._args_from_worker_config(tdef.cmd_args.dynamo.decode_worker))

        decode_cfg["extraPodSpec"]["mainContainer"]["args"] = args

        self._set_multinode_if_needed(decode_cfg, tdef.cmd_args.dynamo.decode_worker)

        return decode_cfg

    def gen_prefill_dict(self, cni_networks: list[str] | None = None) -> dict[str, Any]:
        tdef = cast(AIDynamoTestDefinition, self.test_run.test)
        if not tdef.cmd_args.dynamo.prefill_worker:
            raise ValueError("Prefill worker configuration is not defined in the test definition.")

        prefill_cfg = self._get_base_service_dict(cni_networks)
        prefill_cfg["subComponentType"] = "prefill"
        prefill_cfg["extraPodSpec"]["mainContainer"]["command"] = tdef.cmd_args.dynamo.prefill_worker.cmd.split()

        prefill_cfg["extraPodSpec"]["mainContainer"]["args"] = [
            "--model",
            tdef.cmd_args.dynamo.model,
            "--is-prefill-worker",
            *self._args_from_worker_config(tdef.cmd_args.dynamo.prefill_worker),
        ]

        self._set_multinode_if_needed(prefill_cfg, tdef.cmd_args.dynamo.prefill_worker)

        return prefill_cfg

    def gen_json(self) -> Dict[Any, Any]:
        td = cast(AIDynamoTestDefinition, self.test_run.test)
        k8s_system = cast(KubernetesSystem, self.system)
        cni_networks = k8s_system.resolve_cni_networks()

        deployment = {
            "apiVersion": "nvidia.com/v1alpha1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": k8s_system.default_namespace},
            "spec": {
                "services": {
                    "frontend": self.gen_frontend_dict(cni_networks),
                    "decode": self.gen_decode_dict(cni_networks),
                },
            },
        }
        if td.cmd_args.dynamo.prefill_worker:
            deployment["spec"]["services"]["prefill"] = self.gen_prefill_dict(cni_networks)

        with (self.test_run.output_path / self.DEPLOYMENT_FILE_NAME).open("w") as f:
            yaml.safe_dump(deployment, f)

        return deployment

    def _get_base_service_dict(self, cni_networks: list[str] | None) -> dict[str, Any]:
        system = cast(KubernetesSystem, self.system)
        tdef = cast(AIDynamoTestDefinition, self.test_run.test)
        cfg: dict[str, Any] = {
            "dynamoNamespace": system.default_namespace,
            "componentType": "worker",
            "replicas": 1,
            "resources": {"limits": {"gpu": f"{system.gpus_per_node}"}},
            "extraPodSpec": {
                "mainContainer": {
                    "image": tdef.cmd_args.docker_image_url,
                    "workingDir": tdef.cmd_args.dynamo.workspace_path,
                }
            },
        }
        if cni_networks is not None:
            nic_resources = {f"nvidia.com/{net.split('/', 1)[1]}": "1" for net in cni_networks}
            cfg["extraPodMetadata"] = {"annotations": {"k8s.v1.cni.cncf.io/networks": ",".join(cni_networks)}}
            cfg["resources"]["requests"] = {"custom": nic_resources}
            cfg["resources"]["limits"]["custom"] = nic_resources
        else:
            cfg["extraPodSpec"]["hostNetwork"] = True
        return cfg

    def _to_dynamo_arg(self, arg_name: str) -> str:
        return "--" + arg_name.replace("_", "-")

    def _dynamo_args_dict(self, model: WorkerBaseArgs) -> dict:
        return model.model_dump(exclude={"num_nodes", "extra_args", "nodes"}, exclude_none=True)

    def _args_from_worker_config(self, worker: WorkerConfig) -> list[str]:
        args = []
        for arg, value in self._dynamo_args_dict(worker.args).items():
            args.extend([self._to_dynamo_arg(arg), str(value)])
        if worker.extra_args:
            args.append(f"{worker.extra_args}")
        return args

    def _set_multinode_if_needed(self, cfg: dict[str, Any], worker: WorkerConfig) -> None:
        if cast(int, worker.num_nodes) > 1:
            cfg["multinode"] = {"nodeCount": worker.num_nodes}
