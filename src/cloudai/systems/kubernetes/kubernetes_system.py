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

import asyncio
import logging
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import yaml

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
        """
        Check if a given Kubernetes job is currently running.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is running, False otherwise.
        """
        k_job: KubernetesJob = cast(KubernetesJob, job)
        return self._is_job_running(k_job.name, k_job.kind)

    def is_job_completed(self, job: BaseJob) -> bool:
        """
        Check if a given Kubernetes job is completed.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is completed, False otherwise.
        """
        k_job: KubernetesJob = cast(KubernetesJob, job)
        return not self._is_job_running(k_job.name, k_job.kind)

    def _is_job_running(self, job_name: str, job_kind: str) -> bool:
        """
        Check if a job is currently running.

        Args:
            job_name (str): The name of the job.
            job_kind (str): The kind of the job ('MPIJob' or 'Job').

        Returns:
            bool: True if the job is running, False if the job has completed or is not found.
        """
        logging.debug(f"Checking for job '{job_name}' of kind '{job_kind}' to determine if it is running.")

        if "mpijob" in job_kind.lower():
            return self._is_mpijob_running(job_name)
        elif "job" in job_kind.lower():
            return self._is_batch_job_running(job_name)
        else:
            error_message = (
                f"Unsupported job kind: '{job_kind}'. Supported kinds are 'MPIJob' for MPI workloads and 'Job' for "
                f"batch jobs. Please verify that the 'job_kind' field is correctly set in the job specification."
            )
            logging.error(error_message)
            raise ValueError(error_message)

    def _is_mpijob_running(self, job_name: str) -> bool:
        """
        Check if an MPIJob is currently running.

        Args:
            job_name (str): The name of the MPIJob.

        Returns:
            bool: True if the MPIJob is running, False if the MPIJob has completed or is not found.
        """
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
        """
        Check if a batch job is currently running.

        Args:
            job_name (str): The name of the batch job.

        Returns:
            bool: True if the batch job is running, False if the job has completed or is not found.
        """
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

    def kill(self, job: BaseJob) -> None:
        """
        Terminate a Kubernetes job.

        Args:
            job (BaseJob): The job to be terminated.
        """
        k_job: KubernetesJob = cast(KubernetesJob, job)
        self.delete_job(k_job.name, k_job.kind)

    def delete_job(self, job_name: str, job_kind: str) -> None:
        """
        Delete a job.

        Args:
            job_name (str): The name of the job.
            job_kind (str): The kind of the job ('MPIJob' or 'Job').
        """
        if "mpijob" in job_kind.lower():
            self._delete_mpi_job(job_name)
        elif "job" in job_kind.lower():
            self._delete_batch_job(job_name)
        else:
            error_message = (
                f"Unsupported job kind: '{job_kind}'. Supported kinds are 'MPIJob' for MPI workloads and 'Job' for "
                "batch jobs. Please verify that the 'job_kind' field is correctly set in the job specification."
            )
            logging.error(error_message)
            raise ValueError(error_message)

    def _delete_mpi_job(self, job_name: str) -> None:
        """
        Delete an MPIJob.

        Args:
            job_name (str): The name of the job.
        """
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
        """
        Delete a batch job.

        Args:
            job_name (str): The name of the job.
        """
        logging.debug(f"Deleting batch job '{job_name}'")
        api_response = self.batch_v1.delete_namespaced_job(
            name=job_name,
            namespace=self.default_namespace,
            body=k8s.client.V1DeleteOptions(propagation_policy="Foreground", grace_period_seconds=5),
        )
        api_response = cast(k8s.client.V1Job, api_response)

        logging.debug(f"Batch job '{job_name}' deleted with status: {api_response.status}")

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
        """
        Submit a job.

        Args:
            job_spec (Dict[Any, Any]): The job specification.

        Returns:
            str: The job name.

        Raises:
            ValueError: If the job specification does not contain a valid 'kind' field.
        """
        api_version = job_spec.get("apiVersion", "")
        kind = job_spec.get("kind", "").lower()

        if "mpijob" in kind:
            return self._create_mpi_job(job_spec)
        elif ("batch" in api_version) and ("job" in kind):
            return self._create_batch_job(job_spec)
        else:
            error_message = (
                f"Unsupported job kind: '{job_spec.get('kind')}'.\n"
                "The supported kinds are: 'MPIJob' for MPI workloads and 'Job' for batch jobs.\n"
                "Please review the job specification generation logic to ensure that the 'kind' field is set "
                "correctly.\n"
            )
            logging.error(error_message)
            raise ValueError(error_message)

    def _create_batch_job(self, job_spec: Dict[Any, Any]) -> str:
        """
        Submit a batch job.

        Args:
            job_spec (Dict[Any, Any]): The job specification.

        Returns:
            str: The job name.
        """
        api_response = self.batch_v1.create_namespaced_job(body=job_spec, namespace=self.default_namespace)

        if not isinstance(api_response, lazy.k8s.client.V1Job) or api_response.metadata is None:
            raise ValueError("Job creation failed or returned an unexpected type")

        job_name: str = api_response.metadata.name
        logging.debug(f"Job '{job_name}' created with status: {api_response.status}")
        return job_name

    def _create_mpi_job(self, job_spec: Dict[Any, Any]) -> str:
        """
        Submit an MPIJob.

        Args:
            job_spec (Dict[Any, Any]): The MPIJob specification.

        Returns:
            str: The job name.
        """
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

    def _is_job_observable(self, job_name: str, job_kind: str) -> bool:
        """
        Check if a job is observable by the Kubernetes client.

        Args:
            job_name (str): The name of the job.
            job_kind (str): The kind of the job (e.g., 'Job', 'MPIJob').

        Returns:
            bool: True if the job is observable, False otherwise.
        """
        logging.debug(f"Checking if job '{job_name}' of kind '{job_kind}' is observable.")

        if "mpijob" in job_kind.lower():
            return self._is_mpijob_observable(job_name)
        elif "job" in job_kind.lower():
            return self._is_batch_job_observable(job_name)
        else:
            logging.error(f"Unsupported job kind: '{job_kind}'")
            return False

    def _is_mpijob_observable(self, job_name: str) -> bool:
        """
        Check if an MPIJob is observable by the Kubernetes client.

        Args:
            job_name (str): The name of the MPIJob.

        Returns:
            bool: True if the MPIJob is observable, False otherwise.
        """
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
        """
        Check if a batch job is observable by the Kubernetes client.

        Args:
            job_name (str): The name of the batch job.

        Returns:
            bool: True if the batch job is observable, False otherwise.
        """
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

    async def apply_manifest(self, manifest: Dict[str, Any]) -> str:
        if manifest.get("kind", "").lower() == "mpijob":
            return await self._create_mpi_job(manifest)
        return await self._create_batch_job(manifest)

    async def apply_file(self, file_path: Path) -> None:
        cmd = ["kubectl", "apply", "-f", str(file_path)]
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"kubectl apply failed: {stderr.decode()}")

    async def helm_upgrade(
        self, release: str, chart: Path, namespace: str, values: Dict[str, Any] | None = None
    ) -> None:
        try:
            cmd = ["helm", "upgrade", "--install", release, str(chart), "--namespace", namespace, "--wait", "--atomic"]

            if values:
                with TemporaryDirectory() as tmpdir:
                    values_file = Path(tmpdir) / "values.yaml"
                    values_file.write_text(yaml.dump(values))
                    cmd.extend(["-f", str(values_file)])

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"Helm command failed: {stderr.decode()}")

        except Exception as e:
            raise RuntimeError(f"Helm upgrade failed: {e!s}") from e

    async def wait_for_resource(self, resource: str, selector: str, condition: str, namespace: str) -> None:
        try:
            cmd = [
                "kubectl",
                "wait",
                f"--for=condition={condition}",
                f"{resource}",
                f"-l{selector}",
                f"--namespace={namespace}",
                "--timeout=5m",
            ]
            process = await asyncio.create_subprocess_exec(*cmd)
            await process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"Wait command failed for {resource}")

        except Exception as e:
            raise RuntimeError(f"Wait for resource failed: {e!s}") from e

    async def setup_port_forward(self, pod_selector: str, local_port: int, remote_port: int, namespace: str) -> None:
        try:
            cmd = [
                "kubectl",
                "port-forward",
                f"--namespace={namespace}",
                f"$(kubectl get pods -l {pod_selector} -o name)",
                f"{local_port}:{remote_port}",
            ]
            process = await asyncio.create_subprocess_exec(*cmd)
            await asyncio.sleep(2)  # Give port-forward time to establish

            if process.returncode is not None:
                raise RuntimeError("Port-forward failed to start")

        except Exception as e:
            raise RuntimeError(f"Port-forward setup failed: {e!s}") from e
