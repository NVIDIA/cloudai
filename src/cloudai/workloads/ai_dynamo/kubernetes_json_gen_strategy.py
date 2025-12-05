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

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, cast

import yaml

from cloudai.core import JsonGenStrategy
from cloudai.systems.kubernetes import KubernetesSystem

from .ai_dynamo import AIDynamoTestDefinition, WorkerBaseArgs


class AIDynamoKubernetesJsonGenStrategy(JsonGenStrategy):
    """JSON generation strategy for AI Dynamo on Kubernetes systems."""

    def _install_python_packages(self, repo_root: Path, venv_pip: Path) -> None:
        installs = [
            ("perf_analyzer", repo_root),
            ("genai-perf", repo_root / "genai-perf"),
        ]

        for package, path in installs:
            install_cmd = f"cd {path} && {venv_pip} install ."
            logging.info(f"Installing {package} with command: {install_cmd}")
            subprocess.run(install_cmd, shell=True, capture_output=True, text=True, check=True)

    def _setup_genai(self, td: AIDynamoTestDefinition) -> None:
        python_exec = td.python_executable
        if not python_exec.venv_path:
            raise ValueError(
                f"The virtual environment for git repo {python_exec.git_repo} does not exist. "
                "Please ensure to run installation before running the test."
            )

        venv_pip = python_exec.venv_path.absolute() / "bin" / "pip"
        assert python_exec.git_repo.installed_path
        repo_root = python_exec.git_repo.installed_path.absolute()

        self._install_python_packages(repo_root, venv_pip)

    def gen_frontend_dict(self) -> dict[str, Any]:
        system = cast(KubernetesSystem, self.system)
        tdef = cast(AIDynamoTestDefinition, self.test_run.test)
        return {
            "dynamoNamespace": system.default_namespace,
            "componentType": "frontend",
            "replicas": 1,
            "extraPodSpec": {
                "mainContainer": {
                    "image": tdef.cmd_args.docker_image_url,
                }
            },
        }

    def _to_dynamo_arg(self, arg_name: str) -> str:
        return "--" + arg_name.replace("_", "-")

    def _dynamo_args_dict(self, model: WorkerBaseArgs) -> dict:
        return model.model_dump(exclude={"num_nodes", "extra_args"}, exclude_none=True)

    def args_from_worker_config(self, worker: WorkerBaseArgs) -> list[str]:
        args = []
        for arg, value in self._dynamo_args_dict(worker).items():
            args.extend([self._to_dynamo_arg(arg), str(value)])
        if worker.extra_args:
            args.append(f"{worker.extra_args}")
        return args

    def set_multinode_if_needed(self, cfg: dict[str, Any], worker: WorkerBaseArgs) -> None:
        if cast(int, worker.num_nodes) > 1:
            cfg["multinode"] = {"nodeCount": worker.num_nodes}

    def gen_decode_dict(self) -> dict[str, Any]:
        system = cast(KubernetesSystem, self.system)
        tdef = cast(AIDynamoTestDefinition, self.test_run.test)
        decode_cfg = {
            "dynamoNamespace": system.default_namespace,
            "componentType": "worker",
            "replicas": 1,
            "resources": {"limits": {"gpu": f"{system.gpus_per_node}"}},
            "extraPodSpec": {
                "mainContainer": {
                    "image": tdef.cmd_args.docker_image_url,
                    "workingDir": tdef.cmd_args.dynamo.workspace_path,
                    "command": tdef.cmd_args.dynamo.decode_cmd.split(),
                }
            },
        }

        args = ["--model", tdef.cmd_args.dynamo.model]
        if tdef.cmd_args.dynamo.prefill_worker:
            decode_cfg["subComponentType"] = "decode-worker"
            args.append("--is-decode-worker")
        args.extend(self.args_from_worker_config(tdef.cmd_args.dynamo.decode_worker))

        decode_cfg["extraPodSpec"]["mainContainer"]["args"] = args

        self.set_multinode_if_needed(decode_cfg, tdef.cmd_args.dynamo.decode_worker)

        return decode_cfg

    def gen_prefill_dict(self) -> dict[str, Any]:
        system = cast(KubernetesSystem, self.system)
        tdef = cast(AIDynamoTestDefinition, self.test_run.test)
        if not tdef.cmd_args.dynamo.prefill_worker:
            raise ValueError("Prefill worker configuration is not defined in the test definition.")

        prefill_cfg = {
            "dynamoNamespace": system.default_namespace,
            "componentType": "worker",
            "subComponentType": "prefill",
            "replicas": 1,
            "resources": {"limits": {"gpu": f"{system.gpus_per_node}"}},
            "extraPodSpec": {
                "mainContainer": {
                    "image": tdef.cmd_args.docker_image_url,
                    "workingDir": tdef.cmd_args.dynamo.workspace_path,
                    "command": tdef.cmd_args.dynamo.prefill_cmd.split(),
                }
            },
        }

        prefill_cfg["extraPodSpec"]["mainContainer"]["args"] = [
            "--model",
            tdef.cmd_args.dynamo.model,
            "--is-prefill-worker",
            *self.args_from_worker_config(tdef.cmd_args.dynamo.prefill_worker),
        ]

        self.set_multinode_if_needed(prefill_cfg, tdef.cmd_args.dynamo.prefill_worker)

        return prefill_cfg

    def gen_json(self) -> Dict[Any, Any]:
        td = cast(AIDynamoTestDefinition, self.test_run.test)
        k8s_system = cast(KubernetesSystem, self.system)

        self._setup_genai(td)

        deployment = {
            "apiVersion": "nvidia.com/v1alpha1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": k8s_system.default_namespace},
            "spec": {
                "services": {
                    "Frontend": self.gen_frontend_dict(),
                    "VllmDecodeWorker": self.gen_decode_dict(),
                },
            },
        }
        if td.cmd_args.dynamo.prefill_worker:
            deployment["spec"]["services"]["VllmPrefillWorker"] = self.gen_prefill_dict()

        with (self.test_run.output_path / "deployment.yaml").open("w") as f:
            yaml.safe_dump(deployment, f)

        return deployment
