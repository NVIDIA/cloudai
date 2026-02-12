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
import ssl
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from websockets.sync.client import connect as ws_connect


class RunAIRestClient:
    """
    Client to interact with the RunAI REST API endpoints.

    REST API documentation can be found at https://api-docs.run.ai/latest/
    """

    def __init__(self, base_url: str, app_id: str, app_secret: str) -> None:
        """
        Initialize the client and automatically retrieve the access token.

        To generate `app_id` and `app_secret`, create a new application in the target RunAI cluster with a unique name.
        Upon creation, the Client ID and Client secret will be returned, which correspond to `app_id` and
        `app_secret` respectively.
        """
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.access_token = self._get_access_token()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.access_token}",
            }
        )

    # --- Private utility method ---
    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make an HTTP request and return JSON."""
        url: str = f"{self.base_url}{path}"
        try:
            response = requests.request(method, url, headers=self.session.headers, params=params, json=data)
            response.raise_for_status()
            return response.json() if response.text else None
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
            logging.error(f"Status Code: {http_err.response.status_code}")
            logging.error(f"Response Content: {http_err.response.text}")
            raise
        except Exception as err:
            logging.error(f"An error occurred: {err}")
            raise

    def _get_access_token(self) -> str:
        """Retrieve an access token using AppId and AppSecret."""
        url = f"{self.base_url}/api/v1/token"
        payload = {"grantType": "app_token", "AppId": self.app_id, "AppSecret": self.app_secret}

        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            token = data.get("accessToken")
            if not token:
                raise ValueError("access_token not found in response")
            return token
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred while retrieving access token: {http_err}")
            logging.error(f"Status Code: {http_err.response.status_code}")
            logging.error(f"Response Content: {http_err.response.text}")
            raise
        except Exception as err:
            logging.error(f"An error occurred while retrieving access token: {err}")
            raise

    # ============================ Clusters ============================
    def get_clusters(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get clusters."""
        return self._request("GET", "/api/v1/clusters", params=params)

    def create_cluster(self, cluster_data: Dict[str, Any]) -> Any:
        """Create cluster."""
        return self._request("POST", "/api/v1/clusters", data=cluster_data)

    def get_cluster(self, cluster_uuid: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get cluster by UUID."""
        return self._request("GET", f"/api/v1/clusters/{cluster_uuid}", params=params)

    def update_cluster(self, cluster_uuid: str, cluster_data: Dict[str, Any]) -> Any:
        """Update cluster by UUID."""
        return self._request("PUT", f"/api/v1/clusters/{cluster_uuid}", data=cluster_data)

    def delete_cluster(self, cluster_uuid: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Delete cluster by UUID."""
        return self._request("DELETE", f"/api/v1/clusters/{cluster_uuid}", params=params)

    def get_cluster_metrics(self, cluster_uuid: str, params: Dict[str, Any]) -> Any:
        """Get cluster metrics."""
        return self._request("GET", f"/api/v1/clusters/{cluster_uuid}/metrics", params=params)

    def get_cluster_install_info(self, cluster_uuid: str, params: Dict[str, Any]) -> Any:
        """Get cluster install info."""
        return self._request("GET", f"/api/v1/clusters/{cluster_uuid}/cluster-install-info", params=params)

    # ============================ Node Pools ============================
    def get_node_pools(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get node pools."""
        return self._request("GET", "/api/v1/node-pools", params=params)

    def count_node_pools(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Count node pools."""
        return self._request("GET", "/api/v1/node-pools/count", params=params)

    def create_node_pool(self, payload: Dict[str, Any]) -> Any:
        """Create node pool."""
        return self._request("POST", "/api/v1/node-pools", data=payload)

    def get_node_pool(self, nodepool_id: str) -> Any:
        """Get node pool by ID."""
        return self._request("GET", f"/api/v1/node-pools/{nodepool_id}")

    def delete_node_pool(self, nodepool_id: str) -> Any:
        """Delete node pool."""
        return self._request("DELETE", f"/api/v1/node-pools/{nodepool_id}")

    def update_node_pool(self, nodepool_id: str, payload: Dict[str, Any]) -> Any:
        """Update node pool."""
        return self._request("PUT", f"/api/v1/node-pools/{nodepool_id}", data=payload)

    def patch_node_pool(self, nodepool_id: str, payload: Dict[str, Any]) -> Any:
        """Patch node pool."""
        return self._request("PATCH", f"/api/v1/node-pools/{nodepool_id}", data=payload)

    # ============================ Nodes ============================
    def get_nodes(self, cluster_uuid: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get nodes."""
        return self._request("GET", f"/api/v1/clusters/{cluster_uuid}/nodes", params=params)

    def get_node_telemetry(self, params: Dict[str, Any]) -> Any:
        """Get node telemetry."""
        return self._request("GET", "/api/v1/nodes/telemetry", params=params)

    def get_node_metrics(self, node_id: str, params: Dict[str, Any]) -> Any:
        """Get node metrics."""
        return self._request("GET", f"/api/v1/nodes/{node_id}/metrics", params=params)

    # ============================ Projects ============================
    def get_projects(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get projects."""
        return self._request("GET", "/api/v1/org-unit/projects", params=params)

    def create_project(self, project_data: Dict[str, Any]) -> Any:
        """Create project."""
        return self._request("POST", "/api/v1/org-unit/projects", data=project_data)

    def update_project(self, project_id: str, project_data: Dict[str, Any]) -> Any:
        """Update project."""
        return self._request("PUT", f"/api/v1/org-unit/projects/{project_id}", data=project_data)

    def delete_project(self, project_id: str) -> Any:
        """Delete project."""
        return self._request("DELETE", f"/api/v1/org-unit/projects/{project_id}")

    def get_project_metrics(self, project_id: str, params: Dict[str, Any]) -> Any:
        """Get project metrics."""
        return self._request("GET", f"/api/v1/org-unit/projects/{project_id}/metrics", params=params)

    def update_project_resources(self, project_id: str, payload: Dict[str, Any]) -> Any:
        """Update project resources."""
        return self._request("PUT", f"/api/v1/org-unit/projects/{project_id}/resources", data=payload)

    def patch_project_resources(self, project_id: str, payload: Dict[str, Any]) -> Any:
        """Patch project resources."""
        return self._request("PATCH", f"/api/v1/org-unit/projects/{project_id}/resources", data=payload)

    # ============================ Tenant Settings ============================
    def get_tenant_settings(self) -> Any:
        """Get tenant settings."""
        return self._request("GET", "/v1/k8s/setting")

    def update_tenant_setting(self, key: str, value: Any) -> Any:
        """Update tenant setting."""
        payload: Dict[str, Any] = {"key": key, "value": value}
        return self._request("PUT", "/v1/k8s/setting", data=payload)

    def get_tenant_setting(self, setting_key: str) -> Any:
        """Get tenant setting by key."""
        return self._request("GET", f"/v1/k8s/setting/{setting_key}")

    # ============================ User Applications ============================
    def get_user_applications(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get user applications."""
        return self._request("GET", "/api/v1/user-applications", params=params)

    def create_user_application(self, payload: Dict[str, Any]) -> Any:
        """Create user application."""
        return self._request("POST", "/api/v1/user-applications", data=payload)

    def get_user_application(self, app_id: str) -> Any:
        """Get user application by ID."""
        return self._request("GET", f"/api/v1/user-applications/{app_id}")

    def delete_user_application(self, app_id: str) -> Any:
        """Delete user application."""
        return self._request("DELETE", f"/api/v1/user-applications/{app_id}")

    def regenerate_user_app_secret(self, app_id: str) -> Any:
        """Regenerate user application secret."""
        return self._request("POST", f"/api/v1/user-applications/{app_id}/secret")

    # ============================ Deployments ============================
    def get_deployments(self, cluster_uuid: str) -> Any:
        """Get deployments."""
        return self._request("GET", f"/v1/k8s/clusters/{cluster_uuid}/deployments")

    def get_deployment(self, cluster_uuid: str, deployment_id: str) -> Any:
        """Get deployment by ID."""
        return self._request("GET", f"/v1/k8s/clusters/{cluster_uuid}/deployments/{deployment_id}")

    def create_deployment(self, cluster_uuid: str, payload: Dict[str, Any]) -> Any:
        """Create deployment."""
        return self._request("POST", f"/v1/k8s/clusters/{cluster_uuid}/deployments", data=payload)

    def update_deployment(self, cluster_uuid: str, deployment_id: str, payload: Dict[str, Any]) -> Any:
        """Update deployment."""
        return self._request("PUT", f"/v1/k8s/clusters/{cluster_uuid}/deployments/{deployment_id}", data=payload)

    def delete_deployment(self, cluster_uuid: str, deployment_id: str) -> Any:
        """Delete deployment."""
        return self._request("DELETE", f"/v1/k8s/clusters/{cluster_uuid}/deployments/{deployment_id}")

    # ============================ Events ============================
    def get_workload_events(self, workload_id: str, offset: int = 0, limit: int = 100, sort_order: str = "asc") -> Any:
        """Retrieve workload events and write to a file."""
        params = {
            "offset": offset,
            "limit": limit,
            "sortOrder": sort_order,
        }
        return self._request("GET", f"/api/v1/workloads/{workload_id}/events", params=params)

    # ============================ Pods ============================
    def get_workload_pods(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get workload pods."""
        return self._request("GET", "/api/v1/workloads/pods", params=params)

    def get_workload_pods_count(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get workload pods count."""
        return self._request("GET", "/api/v1/workloads/pods/count", params=params)

    def get_pod_metrics(self, workload_id: str, pod_id: str, params: Dict[str, Any]) -> Any:
        """Get pod metrics."""
        return self._request("GET", f"/api/v1/workloads/{workload_id}/pods/{pod_id}/metrics", params=params)

    # ============================ Trainings ============================
    def create_training(self, payload: Dict[str, Any]) -> Any:
        """Create training."""
        return self._request("POST", "/api/v1/workloads/trainings", data=payload)

    def delete_training(self, workload_id: str) -> Any:
        """Delete training."""
        return self._request("DELETE", f"/api/v1/workloads/trainings/{workload_id}")

    def get_training(self, workload_id: str) -> Any:
        """Get training."""
        return self._request("GET", f"/api/v1/workloads/trainings/{workload_id}")

    def suspend_training(self, workload_id: str) -> Any:
        """Suspend training."""
        return self._request("POST", f"/api/v1/workloads/trainings/{workload_id}/suspend")

    def resume_training(self, workload_id: str) -> Any:
        """Resume training."""
        return self._request("POST", f"/api/v1/workloads/trainings/{workload_id}/resume")

    # ============================ Inferences ============================
    def get_inference(self, workload_id: str) -> Any:
        """Get inference workload."""
        return self._request("GET", f"/api/v1/workloads/inferences/{workload_id}")

    def create_inference(self, payload: Dict[str, Any]) -> Any:
        """Create inference."""
        return self._request("POST", "/api/v1/workloads/inferences", data=payload)

    def update_inference_spec(self, workload_id: str, payload: Dict[str, Any]) -> Any:
        """Update inference spec."""
        return self._request("PATCH", f"/api/v1/workloads/inferences/{workload_id}", data=payload)

    def delete_inference(self, workload_id: str) -> Any:
        """Delete inference."""
        return self._request("DELETE", f"/api/v1/workloads/inferences/{workload_id}")

    def get_inference_metrics(self, workload_id: str, params: Dict[str, Any]) -> Any:
        """Get inference metrics."""
        return self._request("GET", f"/api/v1/workloads/inferences/{workload_id}/metrics", params=params)

    def get_inference_pod_metrics(self, workload_id: str, pod_id: str, params: Dict[str, Any]) -> Any:
        """Get inference pod metrics."""
        return self._request("GET", f"/api/v1/workloads/inferences/{workload_id}/pods/{pod_id}/metrics", params=params)

    # ============================ Storage Classes ============================
    def get_storage_classes_v1(self, cluster_uuid: str, include_none: bool = False) -> Any:
        """Get storage classes (v1)."""
        params: Dict[str, Any] = {"includeNone": include_none}
        return self._request("GET", f"/v1/k8s/clusters/{cluster_uuid}/storage-classes", params=params)

    def get_storage_classes_v2(self, params: Dict[str, Any]) -> Any:
        """Get storage classes (v2)."""
        return self._request("GET", "/api/v2/storage-classes", params=params)

    # ============================ NFS Assets ============================
    def get_nfs_assets(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get NFS assets."""
        return self._request("GET", "/api/v1/asset/datasource/nfs", params=params)

    def create_nfs_asset(self, payload: Dict[str, Any]) -> Any:
        """Create NFS asset."""
        return self._request("POST", "/api/v1/asset/datasource/nfs", data=payload)

    def get_nfs_asset(self, asset_id: str) -> Any:
        """Get NFS asset by ID."""
        return self._request("GET", f"/api/v1/asset/datasource/nfs/{asset_id}")

    def update_nfs_asset(self, asset_id: str, payload: Dict[str, Any]) -> Any:
        """Update NFS asset."""
        return self._request("PUT", f"/api/v1/asset/datasource/nfs/{asset_id}", data=payload)

    def delete_nfs_asset(self, asset_id: str) -> Any:
        """Delete NFS asset."""
        return self._request("DELETE", f"/api/v1/asset/datasource/nfs/{asset_id}")

    # ============================ PVC Assets ============================
    def get_pvc_assets(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get PVC assets."""
        return self._request("GET", "/api/v1/asset/datasource/pvc", params=params)

    def create_pvc_asset(self, payload: Dict[str, Any]) -> Any:
        """Create PVC asset."""
        return self._request("POST", "/api/v1/asset/datasource/pvc", data=payload)

    def get_pvc_asset(self, asset_id: str) -> Any:
        """Get PVC asset by ID."""
        return self._request("GET", f"/api/v1/asset/datasource/pvc/{asset_id}")

    def update_pvc_asset(self, asset_id: str, payload: Dict[str, Any]) -> Any:
        """Update PVC asset."""
        return self._request("PUT", f"/api/v1/asset/datasource/pvc/{asset_id}", data=payload)

    def delete_pvc_asset(self, asset_id: str) -> Any:
        """Delete PVC asset."""
        return self._request("DELETE", f"/api/v1/asset/datasource/pvc/{asset_id}")

    # ============================ Registry Assets ============================
    def get_registry_assets(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get registry assets."""
        return self._request("GET", "/api/v1/asset/registries", params=params)

    def create_registry_asset(self, payload: Dict[str, Any]) -> Any:
        """Create registry asset."""
        return self._request("POST", "/api/v1/asset/registries", data=payload)

    def get_registry_asset(self, asset_id: str) -> Any:
        """Get registry asset by ID."""
        return self._request("GET", f"/api/v1/asset/registries/{asset_id}")

    def update_registry_asset(self, asset_id: str, payload: Dict[str, Any]) -> Any:
        """Update registry asset."""
        return self._request("PUT", f"/api/v1/asset/registries/{asset_id}", data=payload)

    def delete_registry_asset(self, asset_id: str) -> Any:
        """Delete registry asset."""
        return self._request("DELETE", f"/api/v1/asset/registries/{asset_id}")

    def get_registry_repositories(self, asset_id: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get registry repositories."""
        return self._request("GET", f"/api/v1/asset/registries/{asset_id}/repositories", params=params)

    def get_registry_repository_tags(
        self, asset_id: str, repository: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Get registry repository tags."""
        query_params = params or {}
        query_params["repository"] = repository
        return self._request("GET", f"/api/v1/asset/registries/{asset_id}/repositories/tags", params=query_params)

    # ============================ S3 Assets ============================
    def get_s3_assets(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get S3 assets."""
        return self._request("GET", "/api/v1/asset/datasource/s3", params=params)

    def create_s3_asset(self, payload: Dict[str, Any]) -> Any:
        """Create S3 asset."""
        return self._request("POST", "/api/v1/asset/datasource/s3", data=payload)

    def get_s3_asset(self, asset_id: str) -> Any:
        """Get S3 asset by ID."""
        return self._request("GET", f"/api/v1/asset/datasource/s3/{asset_id}")

    def update_s3_asset(self, asset_id: str, payload: Dict[str, Any]) -> Any:
        """Update S3 asset."""
        return self._request("PUT", f"/api/v1/asset/datasource/s3/{asset_id}", data=payload)

    def delete_s3_asset(self, asset_id: str) -> Any:
        """Delete S3 asset."""
        return self._request("DELETE", f"/api/v1/asset/datasource/s3/{asset_id}")

    # ============================ ConfigMap Assets ============================
    def get_configmap_assets(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get ConfigMap assets."""
        return self._request("GET", "/api/v1/asset/datasource/config-map", params=params)

    def create_configmap_asset(self, payload: Dict[str, Any]) -> Any:
        """Create ConfigMap asset."""
        return self._request("POST", "/api/v1/asset/datasource/config-map", data=payload)

    def get_configmap_asset(self, asset_id: str) -> Any:
        """Get ConfigMap asset by ID."""
        return self._request("GET", f"/api/v1/asset/datasource/config-map/{asset_id}")

    def update_configmap_asset(self, asset_id: str, payload: Dict[str, Any]) -> Any:
        """Update ConfigMap asset."""
        return self._request("PUT", f"/api/v1/asset/datasource/config-map/{asset_id}", data=payload)

    def delete_configmap_asset(self, asset_id: str) -> Any:
        """Delete ConfigMap asset."""
        return self._request("DELETE", f"/api/v1/asset/datasource/config-map/{asset_id}")

    # ============================ Secret Assets ============================
    def get_secret_assets(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get Secret assets."""
        return self._request("GET", "/api/v1/asset/datasource/secrets", params=params)

    def create_secret_asset(self, payload: Dict[str, Any]) -> Any:
        """Create Secret asset."""
        return self._request("POST", "/api/v1/asset/datasource/secrets", data=payload)

    def get_secret_asset(self, asset_id: str) -> Any:
        """Get Secret asset by ID."""
        return self._request("GET", f"/api/v1/asset/datasource/secrets/{asset_id}")

    def update_secret_asset(self, asset_id: str, payload: Dict[str, Any]) -> Any:
        """Update Secret asset."""
        return self._request("PUT", f"/api/v1/asset/datasource/secrets/{asset_id}", data=payload)

    def delete_secret_asset(self, asset_id: str) -> Any:
        """Delete Secret asset."""
        return self._request("DELETE", f"/api/v1/asset/datasource/secrets/{asset_id}")

    # ============================ Policies ============================
    def get_training_policy(self, params: Dict[str, Any]) -> Any:
        """Get training policy."""
        return self._request("GET", "/api/v2/policy/trainings", params=params)

    def update_training_policy(self, payload: Dict[str, Any], validate_only: bool = False) -> Any:
        """Update training policy."""
        params: Dict[str, Any] = {"validateOnly": validate_only} if validate_only else {}
        return self._request("PATCH", "/api/v2/policy/trainings", params=params, data=payload)

    def overwrite_training_policy(self, payload: Dict[str, Any], validate_only: bool = False) -> Any:
        """Overwrite training policy."""
        params: Dict[str, Any] = {"validateOnly": validate_only} if validate_only else {}
        return self._request("PUT", "/api/v2/policy/trainings", params=params, data=payload)

    # ============================ Cluster API ============================
    def is_cluster_api_available(self, cluster_domain: str) -> bool:
        url = f"{cluster_domain}/cluster-api/status"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": "runai-cli/2.21.3-saas.4 sdk/2.74.5 go/go1.23.7 darwin/arm64",
            "Accept-Encoding": "gzip",
            "Content-Length": "0",
        }
        response = requests.get(url, headers=headers)
        return "OK" in response.text

    def fetch_training_logs(
        self, cluster_domain: str, project_name: str, training_task_name: str, output_file_path: Path
    ):
        if not self.is_cluster_api_available(cluster_domain):
            logging.error("Cluster API status check failed.")
            return

        cluster_domain = cluster_domain.replace("https://", "wss://")
        url = f"{cluster_domain}/cluster-api/api/v1/{project_name}/workloads/training/runai/{training_task_name}/logs"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": "Go-http-client/1.1",
            "Accept-Encoding": "gzip",
        }

        ssl_context = ssl._create_unverified_context()
        with (
            ws_connect(url, additional_headers=headers, ssl=ssl_context) as websocket,
            output_file_path.open("w") as log_file,
        ):
            for message in websocket:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                log_file.write(str(message))
