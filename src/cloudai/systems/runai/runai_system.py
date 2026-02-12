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
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, PrivateAttr

from cloudai.core import BaseJob, System

from .runai_cluster import RunAICluster
from .runai_event import RunAIEvent
from .runai_node import RunAINode
from .runai_project import RunAIProject
from .runai_pvc import RunAIPVC
from .runai_rest_client import RunAIRestClient
from .runai_training import ActualPhase, RunAITraining


class RunAISystem(System):
    """
    RunAISystem integrates with the RunAI platform to manage and monitor jobs and nodes.

    Attributes:
        name (str): The name of the RunAI system.
        install_path (Path): The installation path for RunAI.
        output_path (Path): The path where output is stored.
        base_url (str): The base URL for the RunAI API.
        app_id (str): The application ID for authentication.
        app_secret (str): The application secret for authentication.
        scheduler (str): The scheduler type, default is "runai".
        global_env_vars (Dict[str, Any]): Global environment variables to be passed to jobs.
        nodes (List[RunAINode]): List of nodes in the RunAI cluster.
    """

    scheduler: str = "runai"
    monitor_interval: int = 60
    base_url: str
    user_email: str
    app_id: str
    app_secret: str
    project_id: str
    cluster_id: str
    nodes: List[RunAINode] = Field(default_factory=list)
    _api_client: Optional[RunAIRestClient] = PrivateAttr(default=None)

    @property
    def api_client(self) -> RunAIRestClient:
        if self._api_client is None:
            self._api_client = RunAIRestClient(
                base_url=self.base_url,
                app_id=self.app_id,
                app_secret=self.app_secret,
            )
        return self._api_client

    @api_client.setter
    def api_client(self, api_client: RunAIRestClient) -> None:
        self._api_client = api_client

    def update(self):
        pass

    def is_job_running(self, job: BaseJob) -> bool:
        """Return True if the specified job is running in the RunAI cluster."""
        training_data = self.api_client.get_training(str(job.id))
        training = RunAITraining(**training_data)
        return training.actual_phase == ActualPhase.RUNNING

    def is_job_completed(self, job: BaseJob) -> bool:
        """Return True if the specified job is completed in the RunAI cluster."""
        training_data = self.api_client.get_training(str(job.id))
        training = RunAITraining(**training_data)
        return training.actual_phase == ActualPhase.COMPLETED

    def kill(self, job: BaseJob) -> None:
        """Terminate a job in the RunAI cluster."""
        response = self.api_client.delete_training(str(job.id))
        if response.status_code == 204:
            logging.debug(f"Job {job.id} successfully terminated.")
        else:
            logging.error(f"Failed to terminate job {job.id}: {response.text}")

    # ============================ Clusters ============================
    def get_clusters(self) -> List[RunAICluster]:
        """Fetch and return clusters as RunAICluster objects."""
        clusters_data = self.api_client.get_clusters()
        return [RunAICluster(**data) for data in clusters_data.get("clusters", [])]

    # ============================ Projects ============================
    def get_projects(self) -> List[RunAIProject]:
        """Fetch and return projects as RunAIProject objects."""
        projects_data = self.api_client.get_projects()
        projects = [RunAIProject(**data) for data in projects_data.get("projects", [])]
        return [project for project in projects if project.created_by == self.user_email]

    def create_project(self, project_data: Dict[str, Any]) -> RunAIProject:
        """Create a project and return it as a RunAIProject object."""
        project_data = self.api_client.create_project(project_data)
        return RunAIProject(**project_data)

    def update_project(self, project_id: str, project_data: Dict[str, Any]) -> RunAIProject:
        """Update a project and return the updated RunAIProject object."""
        updated_data = self.api_client.update_project(project_id, project_data)
        return RunAIProject(**updated_data)

    def delete_project(self, project_id: str) -> None:
        """Delete a project by its ID."""
        self.api_client.delete_project(project_id)

    # ============================ PVC Assets ============================
    def get_pvc_assets(self) -> List[RunAIPVC]:
        """Fetch and return PVC assets as RunAIPVC objects."""
        pvc_data = self.api_client.get_pvc_assets()
        return [RunAIPVC(**pvc) for pvc in pvc_data.get("entries", [])]

    def create_pvc_asset(self, payload: Dict[str, Any]) -> RunAIPVC:
        """Create a PVC asset and return it as a RunAIPVC object."""
        pvc_data = self.api_client.create_pvc_asset(payload)
        return RunAIPVC(**pvc_data)

    def delete_pvc_asset(self, asset_id: str) -> None:
        """Delete a PVC asset by its ID."""
        self.api_client.delete_pvc_asset(asset_id)

    # ============================ Events ============================
    def get_workload_events(
        self, workload_id: str, output_file_path: Path, offset: int = 0, limit: int = 100, sort_order: str = "asc"
    ) -> None:
        """Retrieve workload events and write them to a file."""
        response = self.api_client.get_workload_events(workload_id, offset=offset, limit=limit, sort_order=sort_order)
        events_data = response.get("events", [])

        events: List[RunAIEvent] = [RunAIEvent(**event_data) for event_data in events_data]

        with output_file_path.open("w") as file:
            for event in events:
                file.write(f"{event}\n")

    # ============================ Trainings ============================
    def create_training(self, training_data: Dict[str, Any]) -> RunAITraining:
        """Create a training and return it as a RunAITraining object."""
        training_data = self.api_client.create_training(training_data)
        return RunAITraining(**training_data)

    def delete_training(self, workload_id: str) -> None:
        """Delete a training by its ID."""
        self.api_client.delete_training(workload_id)

    def get_training(self, workload_id: str) -> Any:
        """Get a training by its ID."""
        return self.api_client.get_training(workload_id)

    def suspend_training(self, workload_id: str) -> None:
        """Suspend a training by its ID."""
        self.api_client.suspend_training(workload_id)

    def resume_training(self, workload_id: str) -> None:
        """Resume a training by its ID."""
        self.api_client.resume_training(workload_id)

    # ============================ Logs ============================
    def store_logs(self, workload_id: str, output_file_path: Path):
        """Store logs for a given workload."""
        training_data = self.api_client.get_training(workload_id)
        training = RunAITraining(**training_data)
        cluster_id = training.cluster_id

        projects_data = self.api_client.get_projects()
        projects = [RunAIProject(**data) for data in projects_data.get("projects", [])]
        project = next((p for p in projects if p.id == self.project_id), None)

        if not project:
            logging.error(f"Project with ID {self.project_id} not found.")
            return

        clusters_data = self.api_client.get_clusters()
        clusters = [RunAICluster(**data) for data in clusters_data]
        cluster = next((c for c in clusters if c.uuid == cluster_id), None)

        if not cluster:
            logging.error(f"Cluster with ID {cluster_id} not found.")
            return

        cluster_domain = cluster.domain

        if not cluster_domain:
            logging.error(f"Domain for cluster {cluster_id} not found.")
            return

        self.api_client.fetch_training_logs(cluster_domain, project.name, training.name, output_file_path)
