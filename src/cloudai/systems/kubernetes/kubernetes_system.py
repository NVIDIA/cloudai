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

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

from kubernetes import client, config
from kubernetes.client import ApiException, CustomObjectsApi, V1DeleteOptions, V1Job
from pydantic import BaseModel, ConfigDict

from cloudai import BaseJob, System
from cloudai.runner.kubernetes.kubernetes_job import KubernetesJob


class KubernetesSystem(BaseModel, System):
    """
    Represents a Kubernetes system.

    Attributes
        name (str): The name of the Kubernetes system.
        install_path (Path): Path to the installation directory.
        output_path (Path): Path to the output directory.
        kube_config_path (Path): Path to the Kubernetes config file.
        default_namespace (str): The default Kubernetes namespace for jobs.
        default_image (str): Default Docker image to be used for jobs.
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
    default_image: str
    scheduler: str = "kubernetes"
    global_env_vars: Dict[str, Any] = {}
    monitor_interval: int = 1
    _core_v1: client.CoreV1Api
    _batch_v1: client.BatchV1Api
    _custom_objects_api: CustomObjectsApi

    def __init__(self, **data):
        """Initialize the KubernetesSystem instance."""
        super().__init__(**data)

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
        config.load_kube_config(config_file=str(kube_config_path))

        self._core_v1 = client.CoreV1Api()
        self._batch_v1 = client.BatchV1Api()
        self._custom_objects_api = CustomObjectsApi()

        logging.debug(f"{self.__class__.__name__} initialized")

    @property
    def core_v1(self) -> client.CoreV1Api:
        """Returns the Kubernetes Core V1 API client."""
        return self._core_v1

    @property
    def batch_v1(self) -> client.BatchV1Api:
        """Returns the Kubernetes Batch V1 API client."""
        return self._batch_v1

    @property
    def custom_objects_api(self) -> CustomObjectsApi:
        """Returns the Kubernetes Custom Objects API client."""
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
            f"Default Namespace: {self.default_namespace}\n"
            f"Default Docker Image: {self.default_image}"
        )

    def update(self) -> None:
        """
        Update the system object for a Kubernetes system.

        Currently not implemented for KubernetesSystem.
        """
        pass

    def is_job_running(self, job: BaseJob) -> bool:
        """
        Check if a given Kubernetes job is currently running.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is running, False otherwise.
        """
        k_job: KubernetesJob = cast(KubernetesJob, job)
        return self._is_job_running(k_job.namespace, k_job.name, k_job.kind)

    def is_job_completed(self, job: BaseJob) -> bool:
        """
        Check if a given Kubernetes job is completed.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is completed, False otherwise.
        """
        k_job: KubernetesJob = cast(KubernetesJob, job)
        return not self._is_job_running(k_job.namespace, k_job.name, k_job.kind)

    def _is_job_running(self, job_namespace: str, job_name: str, job_kind: str) -> bool:
        """
        Check if a job is currently running.

        Args:
            job_namespace (str): The namespace of the job.
            job_name (str): The name of the job.
            job_kind (str): The kind of the job ('MPIJob' or 'Job').

        Returns:
            bool: True if the job is running, False if the job has completed or is not found.
        """
        logging.debug(
            f"Checking for job '{job_name}' of kind '{job_kind}' in namespace '{job_namespace}' to determine if "
            "it is running."
        )

        if "mpijob" in job_kind.lower():
            return self._is_mpijob_running(job_namespace, job_name)
        elif "job" in job_kind.lower():
            return self._is_batch_job_running(job_namespace, job_name)
        else:
            error_message = (
                f"Unsupported job kind: '{job_kind}'. Supported kinds are 'MPIJob' for MPI workloads and 'Job' for "
                f"batch jobs. Please verify that the 'job_kind' field is correctly set in the job specification."
            )
            logging.error(error_message)
            raise ValueError(error_message)

    def _is_mpijob_running(self, job_namespace: str, job_name: str) -> bool:
        """
        Check if an MPIJob is currently running.

        Args:
            job_namespace (str): The namespace of the MPIJob.
            job_name (str): The name of the MPIJob.

        Returns:
            bool: True if the MPIJob is running, False if the MPIJob has completed or is not found.
        """
        try:
            mpijob = self.custom_objects_api.get_namespaced_custom_object(
                group="kubeflow.org", version="v2beta1", namespace=job_namespace, plural="mpijobs", name=job_name
            )

            assert isinstance(mpijob, dict)
            status = mpijob.get("status", {})
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
            if any(condition["type"] == "Created" and condition["status"] == "True" for condition in conditions):
                return True

            return False

        except ApiException as e:
            if e.status == 404:
                logging.debug(
                    f"MPIJob '{job_name}' not found in namespace '{job_namespace}'. It may have completed and been "
                    "removed from the system."
                )
                return False
            else:
                error_message = (
                    f"Error occurred while retrieving status for MPIJob '{job_name}' in namespace '{job_namespace}'. "
                    f"Error code: {e.status}. Message: {e.reason}. Please check the job name, namespace, and "
                    "Kubernetes API server."
                )
                logging.error(error_message)
                raise

    def _is_batch_job_running(self, job_namespace: str, job_name: str) -> bool:
        """
        Check if a batch job is currently running.

        Args:
            job_namespace (str): The namespace of the batch job.
            job_name (str): The name of the batch job.

        Returns:
            bool: True if the batch job is running, False if the job has completed or is not found.
        """
        try:
            k8s_job: Any = self.batch_v1.read_namespaced_job_status(name=job_name, namespace=job_namespace)

            if not (hasattr(k8s_job, "status") and hasattr(k8s_job.status, "conditions")):
                logging.debug(
                    f"Job '{job_name}' in namespace '{job_namespace}' does not have expected status attributes."
                )
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

        except ApiException as e:
            if e.status == 404:
                logging.debug(
                    f"Batch job '{job_name}' not found in namespace '{job_namespace}'."
                    "It may have completed and been removed from the system."
                )
                return False
            else:
                logging.error(
                    f"Error occurred while retrieving status for batch job '{job_name}' in namespace "
                    "'{job_namespace}'."
                    f"Error code: {e.status}. Message: {e.reason}. Please check the job name, namespace, "
                    "and Kubernetes API server."
                )
                raise

    def kill(self, job: BaseJob) -> None:
        """
        Terminate a Kubernetes job.

        Args:
            job (BaseJob): The job to be terminated.
        """
        k_job: KubernetesJob = cast(KubernetesJob, job)
        self.delete_job(k_job.namespace, k_job.name, k_job.kind)

    def delete_job(self, namespace: str, job_name: str, job_kind: str) -> None:
        """
        Delete a job in the specified namespace.

        Args:
            namespace (str): The namespace of the job.
            job_name (str): The name of the job.
            job_kind (str): The kind of the job ('MPIJob' or 'Job').
        """
        if "mpijob" in job_kind.lower():
            self._delete_mpi_job(namespace, job_name)
        elif "job" in job_kind.lower():
            self._delete_batch_job(namespace, job_name)
        else:
            error_message = (
                f"Unsupported job kind: '{job_kind}'. Supported kinds are 'MPIJob' for MPI workloads and 'Job' for "
                "batch jobs. Please verify that the 'job_kind' field is correctly set in the job specification."
            )
            logging.error(error_message)
            raise ValueError(error_message)

    def _delete_mpi_job(self, namespace: str, job_name: str) -> None:
        """
        Delete an MPIJob in the specified namespace.

        Args:
            namespace (str): The namespace of the job.
            job_name (str): The name of the job.
        """
        logging.debug(f"Deleting MPIJob '{job_name}' in namespace '{namespace}'")
        try:
            self.custom_objects_api.delete_namespaced_custom_object(
                group="kubeflow.org",
                version="v2beta1",
                namespace=namespace,
                plural="mpijobs",
                name=job_name,
                body=V1DeleteOptions(propagation_policy="Foreground", grace_period_seconds=5),
            )
            logging.debug(f"MPIJob '{job_name}' deleted successfully in namespace '{namespace}'")
        except ApiException as e:
            if e.status == 404:
                logging.debug(
                    f"MPIJob '{job_name}' not found in namespace '{namespace}'. " "It may have already been deleted."
                )
            else:
                logging.error(
                    f"An error occurred while attempting to delete MPIJob '{job_name}' in namespace '{namespace}'. "
                    f"Error code: {e.status}. Message: {e.reason}. Please verify the job name, namespace, "
                    "and Kubernetes API server."
                )
                raise

    def _delete_batch_job(self, namespace: str, job_name: str) -> None:
        """
        Delete a batch job in the specified namespace.

        Args:
            namespace (str): The namespace of the job.
            job_name (str): The name of the job.
        """
        logging.debug(f"Deleting batch job '{job_name}' in namespace '{namespace}'")
        api_response = self.batch_v1.delete_namespaced_job(
            name=job_name,
            namespace=namespace,
            body=V1DeleteOptions(propagation_policy="Foreground", grace_period_seconds=5),
        )
        api_response = cast(V1Job, api_response)

        logging.debug(f"Batch job '{job_name}' deleted with status: {api_response.status}")

    def create_job(self, job_spec: Dict[Any, Any], timeout: int = 60, interval: int = 1) -> Tuple[str, str]:
        """
        Create a job in the Kubernetes system in a blocking manner.

        Args:
            job_spec (Dict[Any, Any]): The job specification.
            timeout (int): The maximum time to wait for the job to be created and observable.
            interval (int): The time to wait between checks, in seconds.

        Returns:
            Tuple[str, str]: The job name and namespace.

        Raises:
            ValueError: If the job specification does not contain a valid 'kind' field.
            TimeoutError: If the job is not observable within the timeout period.
        """
        logging.debug(f"Creating job with spec: {job_spec}")
        job_name, namespace = self._create_job(self.default_namespace, job_spec)

        # Wait for the job to be observable by Kubernetes
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_job_observable(namespace, job_name, job_spec.get("kind", "")):
                logging.debug(f"Job '{job_name}' is now observable in namespace '{namespace}'.")
                return job_name, namespace
            logging.debug(f"Waiting for job '{job_name}' to become observable in namespace '{namespace}'...")
            time.sleep(interval)

        raise TimeoutError(f"Job '{job_name}' in namespace '{namespace}' was not observable within {timeout} seconds.")

    def _create_job(self, namespace: str, job_spec: Dict[Any, Any]) -> Tuple[str, str]:
        """
        Submit a job to the specified namespace.

        Args:
            namespace (str): The namespace where the job will be created.
            job_spec (Dict[Any, Any]): The job specification.

        Returns:
            Tuple[str, str]: The job name and namespace.

        Raises:
            ValueError: If the job specification does not contain a valid 'kind' field.
        """
        api_version = job_spec.get("apiVersion", "")
        kind = job_spec.get("kind", "").lower()

        if "mpijob" in kind:
            return self._create_mpi_job(namespace, job_spec)
        elif ("batch" in api_version) and ("job" in kind):
            return self._create_batch_job(namespace, job_spec)
        else:
            error_message = (
                f"Unsupported job kind: '{job_spec.get('kind')}'.\n"
                "The supported kinds are: 'MPIJob' for MPI workloads and 'Job' for batch jobs.\n"
                "Please review the job specification generation logic to ensure that the 'kind' field is set "
                "correctly.\n"
            )
            logging.error(error_message)
            raise ValueError(error_message)

    def _create_batch_job(self, namespace: str, job_spec: Dict[Any, Any]) -> Tuple[str, str]:
        """
        Submit a batch job to the specified namespace.

        Args:
            namespace (str): The namespace where the job will be created.
            job_spec (Dict[Any, Any]): The job specification.

        Returns:
            Tuple[str, str]: The job name and namespace.
        """
        logging.debug(f"Creating job in namespace '{namespace}'")
        api_response = self.batch_v1.create_namespaced_job(body=job_spec, namespace=namespace)

        if not isinstance(api_response, V1Job) or api_response.metadata is None:
            raise ValueError("Job creation failed or returned an unexpected type")

        job_name: str = api_response.metadata.name
        job_namespace: str = api_response.metadata.namespace
        logging.debug(f"Job '{job_name}' created with status: {api_response.status}")
        return job_name, job_namespace

    def _create_mpi_job(self, namespace: str, job_spec: Dict[Any, Any]) -> Tuple[str, str]:
        """
        Submit an MPIJob to the specified namespace.

        Args:
            namespace (str): The namespace where the MPIJob will be created.
            job_spec (Dict[Any, Any]): The MPIJob specification.

        Returns:
            Tuple[str, str]: The job name and namespace.
        """
        logging.debug(f"Creating MPIJob in namespace '{namespace}'")
        api_response = self.custom_objects_api.create_namespaced_custom_object(
            group="kubeflow.org",
            version="v2beta1",
            namespace=namespace,
            plural="mpijobs",
            body=job_spec,
        )

        job_name: str = api_response["metadata"]["name"]
        job_namespace: str = api_response["metadata"]["namespace"]
        logging.debug(f"MPIJob '{job_name}' created with status: {api_response.get('status')}")
        return job_name, job_namespace

    def _is_job_observable(self, namespace: str, job_name: str, job_kind: str) -> bool:
        """
        Check if a job is observable by the Kubernetes client.

        Args:
            namespace (str): The namespace of the job.
            job_name (str): The name of the job.
            job_kind (str): The kind of the job (e.g., 'Job', 'MPIJob').

        Returns:
            bool: True if the job is observable, False otherwise.
        """
        logging.debug(f"Checking if job '{job_name}' of kind '{job_kind}' in namespace '{namespace}' is observable.")

        if "mpijob" in job_kind.lower():
            return self._is_mpijob_observable(namespace, job_name)
        elif "job" in job_kind.lower():
            return self._is_batch_job_observable(namespace, job_name)
        else:
            logging.error(f"Unsupported job kind: '{job_kind}'")
            return False

    def _is_mpijob_observable(self, namespace: str, job_name: str) -> bool:
        """
        Check if an MPIJob is observable by the Kubernetes client.

        Args:
            namespace (str): The namespace of the MPIJob.
            job_name (str): The name of the MPIJob.

        Returns:
            bool: True if the MPIJob is observable, False otherwise.
        """
        logging.debug(f"Attempting to observe MPIJob '{job_name}' in namespace '{namespace}'.")
        try:
            api_instance = CustomObjectsApi()
            mpijob = api_instance.get_namespaced_custom_object(
                group="kubeflow.org",
                version="v2beta1",
                namespace=namespace,
                plural="mpijobs",
                name=job_name,
            )
            if mpijob:
                logging.debug(f"MPIJob '{job_name}' found in namespace '{namespace}' with details: {mpijob}.")
                return True
            else:
                logging.debug(f"MPIJob '{job_name}' in namespace '{namespace}' is not yet observable.")
                return False
        except ApiException as e:
            if e.status == 404:
                logging.debug(f"MPIJob '{job_name}' not found in namespace '{namespace}'.")
                return False
            else:
                logging.error(
                    f"An error occurred while checking if MPIJob '{job_name}' is observable: {e.reason}. "
                    f"Please check the job name, namespace, and Kubernetes API server."
                )
                raise

    def _is_batch_job_observable(self, namespace: str, job_name: str) -> bool:
        """
        Check if a batch job is observable by the Kubernetes client.

        Args:
            namespace (str): The namespace of the batch job.
            job_name (str): The name of the batch job.

        Returns:
            bool: True if the batch job is observable, False otherwise.
        """
        logging.debug(f"Attempting to observe batch job '{job_name}' in namespace '{namespace}'.")
        try:
            return self.batch_v1.read_namespaced_job_status(name=job_name, namespace=namespace) is not None
        except ApiException as e:
            if e.status == 404:
                logging.debug(f"Batch job '{job_name}' not found in namespace '{namespace}'.")
                return False
            else:
                logging.error(
                    f"An error occurred while checking if batch job '{job_name}' is observable: {e.reason}. "
                    f"Please check the job name, namespace, and Kubernetes API server."
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

    def create_namespace(self, namespace: str) -> None:
        """
        Create a new namespace in the Kubernetes cluster.

        Args:
            namespace (str): The name of the namespace to create.
        """
        body = client.V1Namespace(metadata=client.V1ObjectMeta(name=namespace))
        self.core_v1.create_namespace(body=body)
        logging.debug(f"Namespace '{namespace}' created successfully.")

    def delete_namespace(self, namespace: str) -> None:
        """
        Delete an existing namespace from the Kubernetes cluster.

        Args:
            namespace (str): The name of the namespace to delete.
        """
        self.core_v1.delete_namespace(name=namespace, body=client.V1DeleteOptions())
        logging.debug(f"Namespace '{namespace}' deleted successfully.")

    def list_namespaces(self) -> List[str]:
        """
        List all namespaces in the Kubernetes cluster.

        Returns
            List[str]: A list of namespace names.
        """
        namespaces = self.core_v1.list_namespace().items
        return [ns.metadata.name for ns in namespaces]

    def store_logs_for_job(self, namespace: str, job_name: str, output_dir: Path) -> None:
        """
        Retrieve and store logs for all pods associated with a given job.

        Args:
            namespace (str): The namespace where the job is running.
            job_name (str): The name of the job.
            output_dir (Path): The directory where logs will be saved.
        """
        pod_names = self.get_pod_names_for_job(namespace, job_name)
        if not pod_names:
            logging.warning(f"No pods found for job '{job_name}' in namespace '{namespace}'")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        stdout_file_path = output_dir / "stdout.txt"

        with stdout_file_path.open("w") as stdout_file:
            for pod_name in pod_names:
                try:
                    logs = self.core_v1.read_namespaced_pod_log(name=pod_name, namespace=namespace)

                    log_file_path = output_dir / f"{pod_name}.txt"
                    with log_file_path.open("w") as log_file:
                        log_file.write(logs)
                    logging.info(f"Logs for pod '{pod_name}' saved to '{log_file_path}'")

                    stdout_file.write(logs + "\n")

                except client.ApiException as e:
                    logging.error(f"Error retrieving logs for pod '{pod_name}' in namespace '{namespace}': {e}")

        logging.info(f"All logs concatenated and saved to '{stdout_file_path}'")

    def get_pod_names_for_job(self, namespace: str, job_name: str) -> List[str]:
        """
        Retrieve pod names associated with a given job.

        Args:
            namespace (str): The namespace where the job is running.
            job_name (str): The name of the job.

        Returns:
            List[str]: A list of pod names associated with the job.
        """
        pod_names = []
        try:
            pods = self.core_v1.list_namespaced_pod(namespace=namespace)
            for pod in pods.items:
                if pod.metadata.labels and pod.metadata.labels.get("training.kubeflow.org/job-name") == job_name:
                    pod_names.append(pod.metadata.name)
        except client.ApiException as e:
            logging.error(f"Error retrieving pods for job '{job_name}' in namespace '{namespace}': {e}")
        return pod_names
