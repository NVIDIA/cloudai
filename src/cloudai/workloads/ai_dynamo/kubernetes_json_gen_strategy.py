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

import logging
import subprocess
from typing import TYPE_CHECKING, Any, Dict, cast

import yaml

if TYPE_CHECKING:
    import kubernetes as k8s
from cloudai.core import JobStatusResult, TestRun
from cloudai.systems.kubernetes import KubernetesJob, KubernetesSystem
from cloudai.systems.kubernetes.json_gen_strategy import JsonGenStrategy
from cloudai.util.lazy_imports import lazy

from .ai_dynamo import AIDynamoTestDefinition, WorkerBaseArgs, WorkerConfig


class AIDynamoKubernetesJsonGenStrategy(JsonGenStrategy):
    """JSON generation strategy for AI Dynamo on Kubernetes systems."""

    DEPLOYMENT_FILE_NAME = "deployment.yaml"

    def __init__(self, system: KubernetesSystem, test_run: TestRun) -> None:
        super().__init__(system, test_run)
        self.system = system
        self._genai_perf_completed = False

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

    def gen_decode_dict(self) -> dict[str, Any]:
        tdef = cast(AIDynamoTestDefinition, self.test_run.test)

        decode_cfg = self._get_base_service_dict()
        decode_cfg["extraPodSpec"]["mainContainer"]["command"] = tdef.cmd_args.dynamo.decode_worker.cmd.split()

        args = ["--model", tdef.cmd_args.dynamo.model]
        if tdef.cmd_args.dynamo.prefill_worker:
            decode_cfg["subComponentType"] = "decode-worker"
            args.append("--is-decode-worker")
        args.extend(self._args_from_worker_config(tdef.cmd_args.dynamo.decode_worker))

        decode_cfg["extraPodSpec"]["mainContainer"]["args"] = args

        self._set_multinode_if_needed(decode_cfg, tdef.cmd_args.dynamo.decode_worker)

        return decode_cfg

    def gen_prefill_dict(self) -> dict[str, Any]:
        tdef = cast(AIDynamoTestDefinition, self.test_run.test)
        if not tdef.cmd_args.dynamo.prefill_worker:
            raise ValueError("Prefill worker configuration is not defined in the test definition.")

        prefill_cfg = self._get_base_service_dict()
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

        deployment = {
            "apiVersion": "nvidia.com/v1alpha1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": k8s_system.default_namespace},
            "spec": {
                "services": {
                    "frontend": self.gen_frontend_dict(),
                    "decode": self.gen_decode_dict(),
                },
            },
        }
        if td.cmd_args.dynamo.prefill_worker:
            deployment["spec"]["services"]["prefill"] = self.gen_prefill_dict()

        with (self.test_run.output_path / self.DEPLOYMENT_FILE_NAME).open("w") as f:
            yaml.safe_dump(deployment, f)

        return deployment

    def _get_base_service_dict(self) -> dict[str, Any]:
        system = cast(KubernetesSystem, self.system)
        tdef = cast(AIDynamoTestDefinition, self.test_run.test)
        return {
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

    ### ↓ RunStrategy interface methods ↓ ###
    def start(self) -> KubernetesJob:
        return self.start_job()

    def stop(self) -> None:
        self.delete_job()

    def is_running(self) -> bool:
        if self._genai_perf_completed:
            return False

        job_name = self.gen_json()["metadata"]["name"]

        if self.are_vllm_pods_ready():
            self._run_genai_perf()
            self._genai_perf_completed = True

            for pod_role in {"decode", "prefill", "frontend"}:
                try:
                    pod_name = self._get_dynamo_pod_by_role(pod_role)
                    logging.debug(f"Fetching logs for {pod_role=} {pod_name=}")
                    logs = self.system.core_v1.read_namespaced_pod_log(
                        name=pod_name, namespace=self.system.default_namespace
                    )
                    with (self.test_run.output_path / f"{pod_role}_pod.log").open("w") as f:
                        f.write(logs)
                except Exception as e:
                    logging.debug(f"Error fetching logs for role '{pod_role}': {e}")

            return False

        deployment = cast(
            dict,
            self.custom_objects_api.get_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.system.default_namespace,
                plural="dynamographdeployments",
                name=job_name,
            ),
        )
        status: dict = cast(dict, deployment.get("status", {}))
        return self._check_deployment_conditions(status.get("conditions", []))

    def is_completed(self) -> bool:
        return self._genai_perf_completed

    ### ↑ RunStrategy interface methods ↑ ###

    @property
    def custom_objects_api(self) -> "k8s.client.CustomObjectsApi":
        self._custom_objects_api = lazy.k8s.client.CustomObjectsApi()
        return self._custom_objects_api

    def _create_job(self) -> str:
        return self._create_dynamo_graph_deployment()

    def is_job_observable(self) -> bool:
        return self._is_dynamo_graph_deployment_observable()

    def _create_dynamo_graph_deployment(self) -> str:
        job_spec = self.gen_json()
        logging.debug(f"Attempting to delete existing job='{job_spec['metadata']['name']}' before creation.")
        self._delete_dynamo_graph_deployment()

        logging.debug("Creating DynamoGraphDeployment with spec")
        try:
            api_response = self.custom_objects_api.create_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.system.default_namespace,
                plural="dynamographdeployments",
                body=job_spec,
            )
        except lazy.k8s.client.ApiException as e:
            logging.error(f"An error occurred while creating DynamoGraphDeployment: {e.reason}")
            self._delete_dynamo_graph_deployment()
            raise

        job_name = str(api_response["metadata"]["name"])
        logging.debug(f"DynamoGraphDeployment '{job_name}' created with status: {api_response.get('status')}")
        return job_name

    def _is_dynamo_graph_deployment_observable(self) -> bool:
        job_name = self.gen_json()["metadata"]["name"]
        logging.debug(f"Attempting to observe DynamoGraphDeployment '{job_name}'.")
        try:
            api_instance = self.custom_objects_api
            deployment = api_instance.get_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.system.default_namespace,
                plural="dynamographdeployments",
                name=job_name,
            )
            if deployment:
                logging.debug(f"DynamoGraphDeployment '{job_name}' found with details: {deployment}.")
                return True
            else:
                logging.debug(f"DynamoGraphDeployment '{job_name}' is not yet observable.")
                return False
        except lazy.k8s.client.ApiException as e:
            if e.status == 404:
                logging.debug(f"DynamoGraphDeployment '{job_name}' not found.")
                return False
            else:
                logging.error(
                    f"An error occurred while checking if DynamoGraphDeployment '{job_name}' "
                    f"is observable: {e.reason}. Please check the job name, namespace, and "
                    "Kubernetes API server."
                )
                raise

    def _delete_dynamo_graph_deployment(self) -> None:
        job_name = self.gen_json()["metadata"]["name"]
        logging.debug(f"Deleting DynamoGraphDeployment '{job_name}'")
        cmd = f"kubectl delete dgd {job_name} -n {self.system.default_namespace}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            logging.debug(f"Failed to delete DynamoGraphDeployment: {result.stderr}")

        self._genai_perf_completed = False

    def delete_job(self) -> None:
        self._delete_dynamo_graph_deployment()

    def are_vllm_pods_ready(self) -> bool:
        job_name = self.gen_json()["metadata"]["name"]
        cmd = ["kubectl", "get", "pods", "-n", self.system.default_namespace]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to get pods: {e}")
            return False

        all_ready = True
        vllm_pods_found = False

        for line in result.stdout.splitlines():
            if line.startswith("NAME"):
                continue

            columns = line.split()
            if len(columns) < 3:
                continue

            pod_name = columns[0]
            if job_name not in pod_name:
                continue

            vllm_pods_found = True
            ready_status = columns[1]
            pod_status = columns[2]

            if pod_status == "Terminating":
                logging.debug(f"Pod {pod_name} is terminating")
                return False

            try:
                ready_count, total_count = map(int, ready_status.split("/"))
            except (ValueError, IndexError) as e:
                logging.error(f"Failed to parse ready status '{ready_status}' for pod {pod_name}: {e}")
                return False

            if pod_status == "Running" and ready_count == total_count:
                logging.debug(f"Pod {pod_name} is running and ready ({ready_status})")
            else:
                logging.debug(f"Pod {pod_name} is {pod_status} but not fully ready ({ready_status})")
                all_ready = False

        if not vllm_pods_found:
            logging.debug("No vLLM pods found")
            return False

        return all_ready

    def _run_genai_perf(self) -> None:
        from cloudai.workloads.ai_dynamo.ai_dynamo import AIDynamoTestDefinition

        if not isinstance(self.test_run.test, AIDynamoTestDefinition):
            raise TypeError("Test definition must be an instance of AIDynamoTestDefinition")
        tdef = cast(AIDynamoTestDefinition, self.test_run.test)

        genai_perf_results_path = "/tmp/cloudai/genai-perf"
        frontend_pod = self._get_dynamo_pod_by_role(role="frontend")

        wrapper_script_path = tdef.cmd_args.genai_perf.script.installed_path

        pod_wrapper_path = "/tmp/genai_perf.sh"

        logging.debug(f"Copying wrapper script {wrapper_script_path} to pod {frontend_pod}")
        cp_wrapper_cmd = [
            "kubectl",
            "cp",
            str(wrapper_script_path),
            f"{self.system.default_namespace}/{frontend_pod}:{pod_wrapper_path}",
        ]
        subprocess.run(cp_wrapper_cmd, capture_output=True, text=True, check=True)

        chmod_cmd = ["chmod", "+x", pod_wrapper_path]
        kubectl_exec_cmd = ["kubectl", "exec", "-n", self.system.default_namespace, frontend_pod, "--", *chmod_cmd]
        logging.debug(f"Making wrapper script executable in pod={frontend_pod}")
        try:
            result = subprocess.run(kubectl_exec_cmd, capture_output=True, text=True, timeout=60 * 10, check=True)
            logging.debug(f"chmod exited {result.returncode}: {result.stdout} {result.stderr}")
        except Exception as e:
            logging.debug(f"Error making wrapper script executable in pod '{frontend_pod}': {e}")

        genai_perf_config: list[str] = [
            "--cmd",
            tdef.cmd_args.genai_perf.cmd,
            "--report-name",
            tdef.cmd_args.genai_perf.report_name,
        ]

        extra_args = tdef.cmd_args.genai_perf.extra_args
        if isinstance(extra_args, list):
            extra_args = " ".join(extra_args)
        if extra_args:
            genai_perf_config.extend(["--extra-args", extra_args])

        # Build genai-perf arguments as --key value pairs for parse_genai_perf_args
        genai_perf_cmd_parts: list[str] = []
        if tdef.cmd_args.genai_perf.args:
            for k, v in tdef.cmd_args.genai_perf.args.model_dump(exclude_none=True).items():
                genai_perf_cmd_parts.extend([f"--{k}", str(v)])

        wrapper_cmd = [
            "/bin/bash",
            pod_wrapper_path,
            "--result-dir",
            genai_perf_results_path,
            "--gpus-per-node",
            str(self.system.gpus_per_node or 1),
            "--model",
            tdef.cmd_args.dynamo.model,
            "--url",
            "http://localhost",
            "--port",
            str(tdef.cmd_args.dynamo.port),
            "--endpoint",
            tdef.cmd_args.dynamo.endpoint,
            *genai_perf_config,
            "--",
            *genai_perf_cmd_parts,
        ]

        kubectl_exec_cmd = ["kubectl", "exec", "-n", self.system.default_namespace, frontend_pod, "--", *wrapper_cmd]
        logging.debug(f"Executing genai-perf in pod={frontend_pod} cmd={kubectl_exec_cmd}")
        try:
            result = subprocess.run(kubectl_exec_cmd, capture_output=True, text=True, timeout=60 * 10)
            logging.debug(f"genai-perf exited with code {result.returncode}")
            with (self.test_run.output_path / "genai_perf.log").open("w") as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\nSTDERR:\n")
                    f.write(result.stderr)
        except Exception as e:
            logging.debug(f"Error executing genai-perf command in pod '{frontend_pod}': {e}")

        self._copy_genai_perf_results(frontend_pod, genai_perf_results_path)

    def _copy_genai_perf_results(self, frontend_pod: str, genai_perf_results_path: str) -> None:
        from cloudai.workloads.ai_dynamo.ai_dynamo import AIDynamoTestDefinition

        tdef = cast(AIDynamoTestDefinition, self.test_run.test)
        assert isinstance(tdef, AIDynamoTestDefinition)
        cmd = [
            "kubectl",
            "cp",
            f"{self.system.default_namespace}/{frontend_pod}:{genai_perf_results_path}",
            str(self.test_run.output_path),
        ]
        logging.debug(f"Copying results with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"Error copying results with command: {' '.join(cmd)}: {result.stderr}")
            return

        report_path = self.test_run.output_path / tdef.cmd_args.genai_perf.report_name
        if not report_path.exists():
            logging.error(f"Genai-perf report not found at {report_path}")
            return

        (self.test_run.output_path / tdef.success_marker).touch()
        logging.debug(f"Success marker touched at {self.test_run.output_path / tdef.success_marker}")

    def _get_dynamo_pod_by_role(self, role: str) -> str:
        for pod in self.system.core_v1.list_namespaced_pod(namespace=self.system.default_namespace).items:
            labels = pod.metadata.labels
            logging.debug(f"Found pod: {pod.metadata.name} with labels: {labels}")
            if labels and str(labels.get("nvidia.com/dynamo-component", "")).lower() == role.lower():  # v0.6.x
                return pod.metadata.name
            if labels and str(labels.get("nvidia.com/dynamo-component-type", "")).lower() == role.lower():  # v0.7.x
                return pod.metadata.name
        raise RuntimeError(f"No pod found for the role '{role}'")

    def _check_deployment_conditions(self, conditions: list) -> bool:
        logging.debug(f"Checking deployment conditions: {conditions}")
        if not conditions:
            return True

        for condition in conditions:
            if condition["type"] == "Ready" and condition["status"] == "True":
                return True
            if condition["type"] == "Failed" and condition["status"] == "True":
                return False

        return True
