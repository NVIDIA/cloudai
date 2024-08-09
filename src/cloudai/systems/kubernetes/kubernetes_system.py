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
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

from kubernetes import client, config
from kubernetes.client import ApiException, CustomObjectsApi, V1DeleteOptions, V1Job

from cloudai import BaseJob, OutputType, System
from cloudai.runner.kubernetes.kubernetes_job import KubernetesJob


class KubernetesSystem(System):
    """
    Represents a Kubernetes system.

    Attributes
        install_path (str): Installation path of CloudAI software.
        output_path (str): Directory path for output files.
        default_image (str): Default Docker image to be used for jobs.
        kube_config_path (str): Path to the Kubernetes config file.
        default_namespace (str): The default Kubernetes namespace for jobs.
        global_env_vars (Optional[Dict[str, Any]]): Dictionary containing additional configuration settings for the
            system.
    """

    def __init__(
        self,
        name: str,
        install_path: str,
        output_path: str,
        default_image: str,
        kube_config_path: str,
        default_namespace: str,
        global_env_vars: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a KubernetesSystem instance.

        Args:
            name (str): Name of the Kubernetes system.
            install_path (str): The installation path of CloudAI.
            output_path (str): Path to the output directory.
            default_image (str): Default Docker image to be used for jobs.
            kube_config_path (str): Path to the Kubernetes config file.
            default_namespace (str): The default Kubernetes namespace for jobs.
            global_env_vars (Optional[Dict[str, Any]]): Dictionary containing additional configuration settings for
                the system.
        """
        super().__init__(name, "kubernetes", output_path)
        self.install_path = install_path
        self.default_image = default_image
        self.kube_config_path = kube_config_path
        self.default_namespace = default_namespace
        self.global_env_vars = global_env_vars if global_env_vars is not None else {}

        # Load the Kubernetes configuration
        if not os.path.exists(kube_config_path):
            error_message = (
                f"Kube config file '{kube_config_path}' not found. This file is required to configure the Kubernetes "
                f"environment. Please verify that the file exists at the specified path."
            )
            logging.error(error_message)
            raise FileNotFoundError(error_message)

        logging.debug(f"Loading kube config from: {kube_config_path}")
        config.load_kube_config(config_file=self.kube_config_path)
        self.core_v1 = client.CoreV1Api()
        self.batch_v1 = client.BatchV1Api()
        self.custom_objects_api = CustomObjectsApi()

        logging.debug(f"{self.__class__.__name__} initialized")

    def __repr__(self) -> str:
        """
        Provide a structured string representation of the system.

        Returns
            str: A string that contains the system name and scheduler type.
        """
        return f"System Name: {self.name}\nScheduler Type: {self.scheduler}"

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
        logging.debug(f"Checking if job '{job.get_id()}' is running in namespace '{self.default_namespace}'")
        k_job: KubernetesJob = cast(KubernetesJob, job)
        job_name = k_job.get_id()
        assert isinstance(job_name, str)
        return self._is_job_running(job_name, self.default_namespace)

    def is_job_completed(self, job: BaseJob) -> bool:
        """
        Check if a given Kubernetes job is completed.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is completed, False otherwise.
        """
        k_job: KubernetesJob = cast(KubernetesJob, job)
        job_name = k_job.get_id()
        assert isinstance(job_name, str)
        logging.debug(f"Checking if job '{job.get_id()}' is completed in namespace '{self.default_namespace}'")
        return not self._is_job_running(job_name, self.default_namespace)

    def kill(self, job: BaseJob) -> None:
        """
        Terminate a Kubernetes job.

        Args:
            job (BaseJob): The job to be terminated.
        """
        logging.debug(f"Terminating job '{job.get_id()}' in namespace '{self.default_namespace}'")
        k_job: KubernetesJob = cast(KubernetesJob, job)
        job_name = k_job.get_id()
        assert isinstance(job_name, str)
        self.delete_job(job_name, self.default_namespace)

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

    def create_job(self, job_spec: Dict[Any, Any]) -> Tuple[str, str]:
        """
        Create a job in the Kubernetes system.

        Args:
            job_spec (Dict[Any, Any]): The job specification.

        Returns:
            Tuple[str, str]: The job name and namespace.

        Raises:
            ValueError: If the job specification does not contain a valid 'kind' field.
        """
        logging.debug(f"Creating job with spec: {job_spec}")
        return self._create_job(self.default_namespace, job_spec)

    def list_jobs(self) -> List[Any]:
        """
        List all jobs in the Kubernetes system's default namespace.

        Returns
            List[Any]: A list of jobs in the namespace.
        """
        logging.debug(f"Listing jobs in namespace '{self.default_namespace}'")
        return self.batch_v1.list_namespaced_job(namespace=self.default_namespace).items

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
            raise ValueError(f"Unsupported job kind: {job_spec['kind']}")

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

    def _is_job_running(self, job_name: str, namespace: str) -> bool:
        """
        Check if a job is currently running. If the job has completed and is no longer present, it will return False.

        Args:
            job_name (str): The name of the job.
            namespace (str): The namespace of the job.

        Returns:
            bool: True if the job is running, False if the job has completed or is not found.
        """
        logging.debug(
            f"Initiating check for job '{job_name}' in namespace '{namespace}' to determine its running status."
        )
        try:
            k8s_job = self.batch_v1.read_namespaced_job_status(name=job_name, namespace=namespace)

            if not isinstance(k8s_job, V1Job) or k8s_job.status is None:
                logging.debug(f"Job '{job_name}' in namespace '{namespace}' is not running. It has likely completed.")
                return False

            is_running = k8s_job.status.completion_time is None
            if is_running:
                logging.debug(f"Job '{job_name}' in namespace '{namespace}' is currently running.")
            else:
                logging.debug(f"Job '{job_name}' in namespace '{namespace}' is not running. It has likely completed.")
            return is_running
        except ApiException as e:
            if e.status == 404:
                logging.debug(
                    f"Job '{job_name}' not found in namespace '{namespace}'. "
                    f"It may have already completed and been removed from the system."
                )
                return False
            else:
                logging.error(
                    f"An error occurred while attempting to check the status of job '{job_name}' in namespace "
                    f"'{namespace}'. Error code: {e.status}. Message: {e.reason}. Please verify that the job name "
                    f"and namespace are correct, and that the Kubernetes API server is accessible. If the issue "
                    f"persists, consider reviewing the job's configuration and logs for further details."
                )
                raise

    def delete_job(self, job_name: str, namespace: str) -> None:
        """
        Delete a job in the specified namespace.

        Args:
            job_name (str): The name of the job.
            namespace (str): The namespace of the job.
        """
        logging.debug(f"Deleting job '{job_name}' in namespace '{namespace}'")
        api_response = self.batch_v1.delete_namespaced_job(
            name=job_name,
            namespace=namespace,
            body=V1DeleteOptions(propagation_policy="Foreground", grace_period_seconds=5),
        )
        api_response = cast(V1Job, api_response)  # Explicitly cast to V1Job

        logging.debug(f"Job '{job_name}' deleted with status: {api_response.status}")

    def retrieve_output_streams(
        self, job: BaseJob, output_type: OutputType, multiple_files: bool = False, line_by_line: bool = False
    ) -> Union[Optional[str], Optional[Iterator[str]], Optional[List[Union[str, Iterator[str]]]]]:
        """
        Retrieve output streams (stdout or stderr) for a given job.

        Args:
            job (BaseJob): The job for which to retrieve output.
            output_type (OutputType): The type of output to retrieve.
            multiple_files (bool): Whether to return output from multiple files separately.
            line_by_line (bool): Whether to return the output line by line as an iterator.

        Returns:
            Union[Optional[str], Optional[Iterator[str]], Optional[List[Union[str, Iterator[str]]]]]:
            The output as a string, an iterator for line-by-line reading, or a list of such outputs.
        """
        # Assuming the job ID is used as the pod label to fetch logs
        job_name = job.get_id()
        assert isinstance(job_name, str)

        # Fetch the pods associated with the job
        pods = self.core_v1.list_namespaced_pod(
            namespace=self.default_namespace, label_selector=f"job-name={job_name}"
        ).items

        if not pods:
            logging.error(f"No pods found for job '{job_name}' in namespace '{self.default_namespace}'.")
            return None

        outputs = []
        pod_name = None  # Initialize pod_name to ensure it is defined before any possible exception
        for pod in pods:
            try:
                pod_name = pod.metadata.name
                container_name = pod.spec.containers[0].name  # Assuming the first container is the main one

                # Fetch logs from the pod
                logs = self.core_v1.read_namespaced_pod_log(
                    name=pod_name, namespace=self.default_namespace, container=container_name, _preload_content=False
                ).stream()

                if line_by_line:
                    log_iterator = iter(logs)
                    if multiple_files:
                        outputs.append(log_iterator)
                    else:
                        return log_iterator
                else:
                    log_content = logs.read()
                    if multiple_files:
                        outputs.append(log_content)
                    else:
                        return log_content

            except ApiException as e:
                logging.error(
                    f"Failed to retrieve logs from pod '{pod_name}' in namespace '{self.default_namespace}'. "
                    f"Error code: {e.status}. Message: {e.reason}"
                )
                return None

        return outputs if outputs else None
