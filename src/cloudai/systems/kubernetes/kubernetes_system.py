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

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

if TYPE_CHECKING:
    import kubernetes as k8s

from pydantic import BaseModel, ConfigDict

from cloudai.core import BaseJob, System
from cloudai.util.lazy_imports import lazy

from .kubernetes_job import KubernetesJob


class KubernetesSystem(BaseModel, System):
    """
    Represents a Kubernetes system.

    Attributes
        name (str): The name of the Kubernetes system.
        install_path (Path): Path to the installation directory.
        output_path (Path): Path to the output directory.
        kube_config_path (Path): Path to the Kubernetes config file.
        default_namespace (str): The default Kubernetes namespace for jobs.
        scheduler (str): The scheduler type, default is "kubernetes".
        global_env_vars (Dict[str, Any]): Global environment variables to be passed to jobs.
        monitor_interval (int): Time interval to monitor jobs, in seconds.
        _core_v1 (client.CoreV1Api): Kubernetes Core V1 API client instance.
        _batch_v1 (client.BatchV1Api): Kubernetes Batch V1 API client instance.
        _custom_objects_api (CustomObjectsApi): Kubernetes Custom Objects API client instance.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    install_path: Path
    output_path: Path
    kube_config_path: Path
    default_namespace: str
    scheduler: str = "kubernetes"
    global_env_vars: Dict[str, Any] = {}
    monitor_interval: int = 1
    _core_v1: Optional[k8s.client.CoreV1Api] = None
    _batch_v1: Optional[k8s.client.BatchV1Api] = None
    _custom_objects_api: Optional[k8s.client.CustomObjectsApi] = None
    _port_forward_process = None
    _test_completed: bool = False

    def __getstate__(self) -> dict[str, Any]:
        """Return the state for pickling, excluding non-picklable Kubernetes client objects."""
        state = self.model_dump(exclude={"_core_v1", "_batch_v1", "_custom_objects_api"})
        return state

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> "KubernetesSystem":  # noqa: Vulture
        """
        Create a deep copy of the KubernetesSystem instance.

        Args:
            memo: Dictionary to keep track of objects that have already been copied.

        Returns:
            A new KubernetesSystem instance with reinitialized Kubernetes clients.
        """
        state = self.__getstate__()
        new_instance = KubernetesSystem(**state)
        new_instance.model_post_init(None)
        return new_instance

    def model_post_init(self, __context: Any = None) -> None:  # noqa: Vulture
        """Initialize the KubernetesSystem instance."""
        kube_config_path = self.kube_config_path
        if not kube_config_path.is_file():
            home_directory = Path.home()
            kube_config_path = home_directory / ".kube" / "config"
        else:
            kube_config_path = kube_config_path.resolve()

        if not kube_config_path.exists():
            error_message = (
                f"Kube config file '{kube_config_path}' not found. This file is required to configure the "
                f"Kubernetes environment. Please verify that the file exists at the specified path."
            )
            logging.error(error_message)
            raise FileNotFoundError(error_message)

        # Instantiate Kubernetes APIs
        logging.debug(f"Loading kube config from: {kube_config_path}")
        lazy.k8s.config.load_kube_config(config_file=str(kube_config_path))

        self._core_v1 = lazy.k8s.client.CoreV1Api()
        self._batch_v1 = lazy.k8s.client.BatchV1Api()
        self._custom_objects_api = lazy.k8s.client.CustomObjectsApi()

        logging.debug(f"{self.__class__.__name__} initialized")

    @property
    def core_v1(self) -> k8s.client.CoreV1Api:
        """Returns the Kubernetes Core V1 API client."""
        assert self._core_v1 is not None
        return self._core_v1

    @property
    def batch_v1(self) -> k8s.client.BatchV1Api:
        """Returns the Kubernetes Batch V1 API client."""
        assert self._batch_v1 is not None
        return self._batch_v1

    @property
    def custom_objects_api(self) -> k8s.client.CustomObjectsApi:
        """Returns the Kubernetes Custom Objects API client."""
        assert self._custom_objects_api is not None
        return self._custom_objects_api

    def __repr__(self) -> str:
        """
        Provide a structured string representation of the system.

        Returns
            str: A string that contains the system name, scheduler type, kube config path, namespace, and image.
        """
        return (
            f"System Name: {self.name}\n"
            f"Scheduler Type: {self.scheduler}\n"
            f"Kube Config Path: {self.kube_config_path}\n"
            f"Default Namespace: {self.default_namespace}"
        )

    def update(self) -> None:
        """
        Update the system object for a Kubernetes system.

        Currently not implemented for KubernetesSystem.
        """
        pass

    def is_job_running(self, job: BaseJob) -> bool:
        k_job: KubernetesJob = cast(KubernetesJob, job)
        return self._is_job_running(k_job.name, k_job.kind)

    def is_job_completed(self, job: BaseJob) -> bool:
        k_job: KubernetesJob = cast(KubernetesJob, job)
        return not self._is_job_running(k_job.name, k_job.kind)

    def _is_job_running(self, job_name: str, job_kind: str) -> bool:
        logging.debug(f"Checking for job '{job_name}' of kind '{job_kind}' to determine if it is running.")

        if "mpijob" in job_kind.lower():
            return self._is_mpijob_running(job_name)
        elif "job" in job_kind.lower():
            return self._is_batch_job_running(job_name)
        elif "dynamographdeployment" in job_kind.lower():
            return self._is_dynamo_graph_deployment_running(job_name)
        else:
            error_message = f"Unsupported job kind: '{job_kind}'."
            logging.error(error_message)
            raise ValueError(error_message)

    def _is_mpijob_running(self, job_name: str) -> bool:
        try:
            mpijob = self.custom_objects_api.get_namespaced_custom_object(
                group="kubeflow.org",
                version="v2beta1",
                namespace=self.default_namespace,
                plural="mpijobs",
                name=job_name,
            )

            assert isinstance(mpijob, dict)
            status: dict = cast(dict, mpijob.get("status", {}))
            conditions = status.get("conditions", [])

            # Consider an empty conditions list as running
            if not conditions:
                return True

            for condition in conditions:
                if condition["type"] == "Succeeded" and condition["status"] == "True":
                    return False
                if condition["type"] == "Failed" and condition["status"] == "True":
                    return False

            # If the job has been created but is neither succeeded nor failed, it is considered running
            return any(condition["type"] == "Created" and condition["status"] == "True" for condition in conditions)

        except lazy.k8s.client.ApiException as e:
            if e.status == 404:
                logging.debug(f"MPIJob '{job_name}' not found. It may have completed and been removed from the system.")
                return False
            else:
                error_message = (
                    f"Error occurred while retrieving status for MPIJob '{job_name}' "
                    f"Error code: {e.status}. Message: {e.reason}. Please check the job name, namespace, and "
                    "Kubernetes API server."
                )
                logging.error(error_message)
                raise

    def _is_batch_job_running(self, job_name: str) -> bool:
        try:
            k8s_job: Any = self.batch_v1.read_namespaced_job_status(name=job_name, namespace=self.default_namespace)

            if not (hasattr(k8s_job, "status") and hasattr(k8s_job.status, "conditions")):
                logging.debug(f"Job '{job_name}' does not have expected status attributes.")
                return False

            conditions = k8s_job.status.conditions or []

            # Consider an empty conditions list as running
            if not conditions:
                return True

            for condition in conditions:
                if condition.type == "Complete" and condition.status == "True":
                    return False
                if condition.type == "Failed" and condition.status == "True":
                    return False

            return any(condition.type == "Created" and condition.status == "True" for condition in conditions)

        except lazy.k8s.client.ApiException as e:
            if e.status == 404:
                logging.debug(
                    f"Batch job '{job_name}' not found.It may have completed and been removed from the system."
                )
                return False
            else:
                logging.error(
                    f"Error occurred while retrieving status for batch job '{job_name}'."
                    f"Error code: {e.status}. Message: {e.reason}. Please check the job name and Kubernetes API server."
                )
                raise

    def _check_vllm_pods_status(self) -> bool:
        try:
            cmd = f"kubectl get pods -n {self.default_namespace} | grep 'vllm-v1-agg'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                logging.warning("No vLLM pods found")
                return False

            all_ready = True
            for line in result.stdout.splitlines():
                columns = line.split()
                if len(columns) < 3:
                    continue

                pod_name = columns[0]
                ready_status = columns[1]
                pod_status = columns[2]

                if pod_status == "Terminating":
                    return False

                ready_count, total_count = map(int, ready_status.split("/"))
                if pod_status == "Running" and ready_count == total_count:
                    logging.info(f"Pod {pod_name} is running and ready ({ready_status})")
                else:
                    logging.info(f"Pod {pod_name} is {pod_status} but not fully ready ({ready_status})")
                    all_ready = False

            return all_ready

        except subprocess.SubprocessError as e:
            logging.error(f"Error running kubectl command: {e}")
            raise

    def _setup_port_forward(self) -> None:
        if self._port_forward_process and self._port_forward_process.poll() is None:
            logging.info("Port forwarding is already running")
            return

        if not self._check_vllm_pods_status():
            logging.info("Pods are not ready yet, skipping port forward")
            return

        try:
            get_pod_cmd = (
                f"kubectl get pods -n {self.default_namespace} --no-headers | "
                "grep vllm-v1-agg-frontend | "
                "awk 'NR==1{print $1}'"
            )
            cmd = f"kubectl port-forward pod/$({get_pod_cmd}) 8000:8000 -n {self.default_namespace}"
            logging.info("Starting port forwarding")
            self._port_forward_process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            logging.info("Port forwarding started")
        except subprocess.SubprocessError as e:
            logging.error(f"Error setting up port forward: {e}")
            raise

    def _check_model_server(self) -> bool:
        try:
            cmd = "curl -s http://localhost:8000/v1/models"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                logging.warning("Failed to connect to model server")
                return False

            try:
                response = json.loads(result.stdout)
                if response.get("data") and len(response["data"]) > 0:
                    logging.info(f"Model server is running. Response: {result.stdout}")
                    return True
                else:
                    logging.info("Model server is up but no models are loaded yet")
                    return False
            except json.JSONDecodeError:
                logging.warning("Invalid JSON response from model server")
                return False

        except subprocess.SubprocessError as e:
            logging.error(f"Error checking model server: {e}")
            return False

    def _test_chat_completion(self) -> None:
        try:
            cmd = """curl -N -X POST http://localhost:8000/v1/chat/completions \\
                -H 'accept: application/json' \\
                -H 'Content-Type: application/json' \\
                -d '{
                    "model": "Qwen/Qwen3-0.6B",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello! How are you?"
                        }
                    ],
                    "max_tokens": 64,
                    "stream": true,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.2,
                    "top_k": 5
                }'"""

            logging.info("Running chat completion test")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                raise subprocess.SubprocessError(f"Chat completion test failed: {result.stderr}")

            lines = [line for line in result.stdout.splitlines() if line.strip()]

            for line in lines:
                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                            content = chunk["choices"][0]["delta"]["content"]
                            logging.info(f"Response chunk: {content}")
                    except json.JSONDecodeError:
                        logging.warning(f"Failed to parse line: {line}")

        except subprocess.SubprocessError as e:
            logging.error(str(e))
            raise

    def _check_deployment_conditions(self, conditions: list) -> bool:
        if not conditions:
            return True

        for condition in conditions:
            if condition["type"] == "Ready" and condition["status"] == "True":
                return True
            if condition["type"] == "Failed" and condition["status"] == "True":
                return False

        return True

    def _is_dynamo_graph_deployment_running(self, job_name: str) -> bool:
        try:
            if self._test_completed:
                return False

            if self._check_vllm_pods_status():
                self._setup_port_forward()
                if self._port_forward_process and self._check_model_server():
                    logging.info("vLLM server is up and models are loaded")
                    self._test_chat_completion()
                    self._test_completed = True
                    return False

            deployment = self.custom_objects_api.get_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.default_namespace,
                plural="dynamographdeployments",
                name=job_name,
            )

            assert isinstance(deployment, dict)
            status: dict = cast(dict, deployment.get("status", {}))
            return self._check_deployment_conditions(status.get("conditions", []))

        except lazy.k8s.client.ApiException as e:
            if e.status == 404:
                logging.debug(f"DynamoGraphDeployment '{job_name}' not found.")
                return False
            else:
                logging.error(
                    f"Error occurred while retrieving status for DynamoGraphDeployment '{job_name}'. "
                    f"Error code: {e.status}. Message: {e.reason}."
                )
                raise

    def kill(self, job: BaseJob) -> None:
        """
        Terminate a Kubernetes job.

        Args:
            job (BaseJob): The job to be terminated.
        """
        k_job: KubernetesJob = cast(KubernetesJob, job)
        self.delete_job(k_job.name, k_job.kind)

    def delete_job(self, job_name: str, job_kind: str) -> None:
        if "mpijob" in job_kind.lower():
            self._delete_mpi_job(job_name)
        elif "job" in job_kind.lower():
            self._delete_batch_job(job_name)
        elif "dynamographdeployment" in job_kind.lower():
            pass
        else:
            error_message = f"Unsupported job kind: '{job_kind}'."
            logging.error(error_message)
            raise ValueError(error_message)

    def _delete_mpi_job(self, job_name: str) -> None:
        logging.debug(f"Deleting MPIJob '{job_name}'")
        try:
            self.custom_objects_api.delete_namespaced_custom_object(
                group="kubeflow.org",
                version="v2beta1",
                namespace=self.default_namespace,
                plural="mpijobs",
                name=job_name,
                body=lazy.k8s.client.V1DeleteOptions(propagation_policy="Foreground", grace_period_seconds=5),
            )
            logging.debug(f"MPIJob '{job_name}' deleted successfully")
        except lazy.k8s.client.ApiException as e:
            if e.status == 404:
                logging.debug(f"MPIJob '{job_name}' not found. It may have already been deleted.")
            else:
                logging.error(
                    f"An error occurred while attempting to delete MPIJob '{job_name}'. "
                    f"Error code: {e.status}. Message: {e.reason}. "
                    "Please verify the job name and Kubernetes API server."
                )
                raise

    def _delete_batch_job(self, job_name: str) -> None:
        logging.debug(f"Deleting batch job '{job_name}'")
        api_response = self.batch_v1.delete_namespaced_job(
            name=job_name,
            namespace=self.default_namespace,
            body=k8s.client.V1DeleteOptions(propagation_policy="Foreground", grace_period_seconds=5),
        )
        api_response = cast(k8s.client.V1Job, api_response)

        logging.debug(f"Batch job '{job_name}' deleted with status: {api_response.status}")

    def _delete_dynamo_graph_deployment(self, job_name: str) -> None:
        logging.debug(f"Deleting DynamoGraphDeployment '{job_name}'")
        try:
            cmd = f"kubectl delete dgd vllm-v1-agg -n {self.default_namespace}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                raise subprocess.SubprocessError(f"Failed to delete DynamoGraphDeployment: {result.stderr}")
            logging.debug("DynamoGraphDeployment deleted successfully")
        except subprocess.SubprocessError as e:
            logging.error(str(e))
            raise

    def create_job(self, job_spec: Dict[Any, Any], timeout: int = 60, interval: int = 1) -> str:
        """
        Create a job in the Kubernetes system in a blocking manner.

        Args:
            job_spec (Dict[Any, Any]): The job specification.
            timeout (int): The maximum time to wait for the job to be created and observable.
            interval (int): The time to wait between checks, in seconds.

        Returns:
            str: The job name.

        Raises:
            ValueError: If the job specification does not contain a valid 'kind' field.
            TimeoutError: If the job is not observable within the timeout period.
        """
        logging.debug(f"Creating job with spec: {job_spec}")
        job_name = self._create_job(job_spec)

        # Wait for the job to be observable by Kubernetes
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_job_observable(job_name, job_spec.get("kind", "")):
                logging.debug(f"Job '{job_name}' is now observable.")
                return job_name
            logging.debug(f"Waiting for job '{job_name}' to become observable...")
            time.sleep(interval)

        raise TimeoutError(f"Job '{job_name}' was not observable within {timeout} seconds.")

    def _create_job(self, job_spec: Dict[Any, Any]) -> str:
        api_version = job_spec.get("apiVersion", "")
        kind = job_spec.get("kind", "").lower()

        if "mpijob" in kind:
            return self._create_mpi_job(job_spec)
        elif ("batch" in api_version) and ("job" in kind):
            return self._create_batch_job(job_spec)
        elif "dynamographdeployment" in kind:
            return self._create_dynamo_graph_deployment(job_spec)
        else:
            error_message = (
                f"Unsupported job kind: '{job_spec.get('kind')}'.\n"
                "Please review the job specification generation logic to ensure that the 'kind' field is set "
                "correctly.\n"
            )
            logging.error(error_message)
            raise ValueError(error_message)

    def _create_batch_job(self, job_spec: Dict[Any, Any]) -> str:
        api_response = self.batch_v1.create_namespaced_job(body=job_spec, namespace=self.default_namespace)

        if not isinstance(api_response, lazy.k8s.client.V1Job) or api_response.metadata is None:
            raise ValueError("Job creation failed or returned an unexpected type")

        job_name: str = api_response.metadata.name
        logging.debug(f"Job '{job_name}' created with status: {api_response.status}")
        return job_name

    def _create_mpi_job(self, job_spec: Dict[Any, Any]) -> str:
        api_response = self.custom_objects_api.create_namespaced_custom_object(
            group="kubeflow.org",
            version="v2beta1",
            namespace=self.default_namespace,
            plural="mpijobs",
            body=job_spec,
        )

        job_name: str = api_response["metadata"]["name"]
        logging.debug(f"MPIJob '{job_name}' created with status: {api_response.get('status')}")
        return job_name

    def _create_dynamo_graph_deployment(self, job_spec: Dict[Any, Any]) -> str:
        api_response = self.custom_objects_api.create_namespaced_custom_object(
            group="nvidia.com",
            version="v1alpha1",
            namespace=self.default_namespace,
            plural="dynamographdeployments",
            body=job_spec,
        )

        job_name: str = api_response["metadata"]["name"]
        logging.debug(f"DynamoGraphDeployment '{job_name}' created with status: {api_response.get('status')}")
        return job_name

    def _is_job_observable(self, job_name: str, job_kind: str) -> bool:
        logging.debug(f"Checking if job '{job_name}' of kind '{job_kind}' is observable.")

        if "mpijob" in job_kind.lower():
            return self._is_mpijob_observable(job_name)
        elif "job" in job_kind.lower():
            return self._is_batch_job_observable(job_name)
        elif "dynamographdeployment" in job_kind.lower():
            return self._is_dynamo_graph_deployment_observable(job_name)
        else:
            logging.error(f"Unsupported job kind: '{job_kind}'")
            return False

    def _is_mpijob_observable(self, job_name: str) -> bool:
        logging.debug(f"Attempting to observe MPIJob '{job_name}'.")
        try:
            api_instance = self.custom_objects_api
            mpijob = api_instance.get_namespaced_custom_object(
                group="kubeflow.org",
                version="v2beta1",
                namespace=self.default_namespace,
                plural="mpijobs",
                name=job_name,
            )
            if mpijob:
                logging.debug(f"MPIJob '{job_name}' found with details: {mpijob}.")
                return True
            else:
                logging.debug(f"MPIJob '{job_name}' is not yet observable.")
                return False
        except lazy.k8s.client.ApiException as e:
            if e.status == 404:
                logging.debug(f"MPIJob '{job_name}' not found.")
                return False
            else:
                logging.error(
                    f"An error occurred while checking if MPIJob '{job_name}' is observable: {e.reason}. "
                    f"Please check the job name, namespace, and Kubernetes API server."
                )
                raise

    def _is_batch_job_observable(self, job_name: str) -> bool:
        logging.debug(f"Attempting to observe batch job '{job_name}'.")
        try:
            return self.batch_v1.read_namespaced_job_status(name=job_name, namespace=self.default_namespace) is not None
        except lazy.k8s.client.ApiException as e:
            if e.status == 404:
                logging.debug(f"Batch job '{job_name}' not found.")
                return False
            else:
                logging.error(
                    f"An error occurred while checking if batch job '{job_name}' is observable: {e.reason}. "
                    f"Please check the job name, namespace, and Kubernetes API server."
                )
                raise

    def _is_dynamo_graph_deployment_observable(self, job_name: str) -> bool:
        logging.debug(f"Attempting to observe DynamoGraphDeployment '{job_name}'.")
        try:
            api_instance = self.custom_objects_api
            deployment = api_instance.get_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.default_namespace,
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

    def list_jobs(self) -> List[Any]:
        """
        List all jobs in the Kubernetes system's default namespace.

        Returns
            List[Any]: A list of jobs in the namespace.
        """
        logging.debug(f"Listing jobs in namespace '{self.default_namespace}'")
        return self.batch_v1.list_namespaced_job(namespace=self.default_namespace).items

    def create_node_group(self, name: str, node_list: List[str]) -> None:
        """
        Create a node group in the Kubernetes system.

        Args:
            name (str): The name of the node group.
            node_list (List[str]): List of node names to be included in the group.
        """
        logging.debug(f"Creating node group '{name}' with nodes: {node_list}")
        for node in node_list:
            body = {"metadata": {"labels": {"cloudai/node-group": name}}}
            logging.debug(f"Labeling node '{node}' with group '{name}'")
            self.core_v1.patch_node(node, body)

    def store_logs_for_job(self, job_name: str, output_dir: Path) -> None:
        """
        Retrieve and store logs for all pods associated with a given job.

        Args:
            job_name (str): The name of the job.
            output_dir (Path): The directory where logs will be saved.
        """
        pod_names = self.get_pod_names_for_job(job_name)
        if not pod_names:
            logging.warning(f"No pods found for job '{job_name}'")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        stdout_file_path = output_dir / "stdout.txt"

        with stdout_file_path.open("w") as stdout_file:
            for pod_name in pod_names:
                try:
                    logs = self.core_v1.read_namespaced_pod_log(name=pod_name, namespace=self.default_namespace)

                    log_file_path = output_dir / f"{pod_name}.txt"
                    with log_file_path.open("w") as log_file:
                        log_file.write(logs)
                    logging.info(f"Logs for pod '{pod_name}' saved to '{log_file_path}'")

                    stdout_file.write(logs + "\n")

                except lazy.k8s.client.ApiException as e:
                    logging.error(f"Error retrieving logs for pod '{pod_name}': {e}")

        logging.info(f"All logs concatenated and saved to '{stdout_file_path}'")

    def get_pod_names_for_job(self, job_name: str) -> List[str]:
        """
        Retrieve pod names associated with a given job.

        Args:
            job_name (str): The name of the job.

        Returns:
            List[str]: A list of pod names associated with the job.
        """
        pod_names = []
        try:
            pods = self.core_v1.list_namespaced_pod(namespace=self.default_namespace)
            for pod in pods.items:
                if pod.metadata.labels and pod.metadata.labels.get("training.kubeflow.org/job-name") == job_name:
                    pod_names.append(pod.metadata.name)
        except lazy.k8s.client.ApiException as e:
            logging.error(f"Error retrieving pods for job '{job_name}': {e}")
        return pod_names
