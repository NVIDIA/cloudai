# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from pathlib import Path
from typing import ClassVar, List, cast

from cloudai.core import CommandGenStrategy

from .ai_dynamo import AIDynamoTestDefinition


class AIDynamoKubernetesCommandGenStrategy(CommandGenStrategy):
    """Command generation strategy for AI Dynamo on Slurm systems."""

    REQUIRED_ENV_VARS: ClassVar[List[str]] = [
        "DOCKER_SERVER",
        "DOCKER_USERNAME",
        "DOCKER_PASSWORD",
        "IMAGE_TAG",
        "NAMESPACE",
    ]
    POD_PREFIXES: ClassVar[List[str]] = [
        "dynamo-platform-dynamo-operator-controller-manager",
        "dynamo-platform-etcd",
        "dynamo-platform-nats",
        "dynamo-platform-nats-box",
    ]
    MONITOR_TIMEOUT_SECONDS: ClassVar[int] = 180
    MONITOR_INTERVAL_SECONDS: ClassVar[int] = 5

    def _get_dynamo_repo_path(self) -> Path:
        td = cast(AIDynamoTestDefinition, self.test_run.test.test_definition)
        if td.dynamo_repo.installed_path is None:
            raise ValueError("dynamo_repo.installed_path is not set - repo may not be installed")
        return td.dynamo_repo.installed_path.absolute()

    def _prepare_output_paths(self) -> tuple[Path, Path]:
        self.test_run.output_path.mkdir(parents=True, exist_ok=True)
        return (
            self.test_run.output_path / "stdout.txt",
            self.test_run.output_path / "stderr.txt",
        )

    def _validate_and_get_env_vars(self) -> list[str]:
        merged_env = self.system.global_env_vars.copy()
        merged_env.update(self.test_run.test.extra_env_vars)

        missing_vars = [var for var in self.REQUIRED_ENV_VARS if var not in merged_env]
        if missing_vars:
            raise ValueError(f"Required environment variables not set: {', '.join(missing_vars)}")

        return [f"export {k}={v}" for k, v in merged_env.items()]

    def _get_pod_deploy_commands(self, deploy_script: Path) -> list[str]:
        dynamo_repo_root = self._get_dynamo_repo_path()
        deployment_file = "deploy/cloud/helm/platform/components/operator/templates/deployment.yaml"
        return [
            # First check and modify the deployment file if needed
            f"cd {dynamo_repo_root} && if grep -q '.Values.dynamo.groveTerminationDelay' {deployment_file}; then "
            f"sed -i.bak '/^.*\\.Values\\.dynamo\\.groveTerminationDelay/,+2d' {deployment_file}; "
            f"fi",
            # Then proceed with deployment
            "kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -",
            f"cd $(dirname {deploy_script}) && {deploy_script} --crds",
        ]

    def _generate_pod_monitor_script(self) -> str:
        pod_prefixes_str = '"\n    "'.join(self.POD_PREFIXES)

        return f'''
timeout={self.MONITOR_TIMEOUT_SECONDS}
interval={self.MONITOR_INTERVAL_SECONDS}
required_pods=(
    "{pod_prefixes_str}"
)

start_time=$(date +%s)
while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -ge $timeout ]; then
        echo "Timeout waiting for pods to be ready"
        exit 1
    fi

    all_ready=true
    for pod_prefix in "${{required_pods[@]}}"; do
        # Check if any pod with this prefix exists and is running
        matching_pods=$(kubectl get pods -n $NAMESPACE | grep "$pod_prefix" || true)
        if [ -z "$matching_pods" ]; then
            all_ready=false
            echo "No pods found matching $pod_prefix"
            break
        fi
        # Check if at least one pod is Running
        if ! echo "$matching_pods" | awk '{{print $3}}' | grep -q "^Running$"; then
            all_ready=false
            current_status=$(echo "$matching_pods" | awk '{{print $3}}' | tr '\n' ',' | sed 's/,$//')
            echo "Waiting for $pod_prefix (current: $current_status, expected: Running)"
            break
        fi
    done

    if $all_ready; then
        echo "All pods are ready!"
        break
    fi

    sleep $interval
done
'''

    def _write_pod_monitor_script(self, script_content: str) -> Path:
        script_path = (self.test_run.output_path / "monitor_pods.sh").absolute()
        with script_path.open("w") as f:
            f.write(script_content)
        return script_path

    def _get_graph_deploy_commands(self) -> list[str]:
        td = cast(AIDynamoTestDefinition, self.test_run.test.test_definition)
        if not hasattr(td.cmd_args, "dynamo_graph_path"):
            raise ValueError("dynamo_graph_path not found in cmd_args")

        graph_path = Path(td.cmd_args.dynamo_graph_path)
        if not graph_path.exists():
            raise ValueError(f"Graph file not found at {graph_path}")

        return [
            f"kubectl apply -f {graph_path.absolute()} -n $NAMESPACE",
        ]

    def gen_exec_command(self) -> str:
        deploy_script = self._get_dynamo_repo_path() / "deploy/cloud/helm/deploy.sh"
        stdout_path, stderr_path = self._prepare_output_paths()
        env_vars = self._validate_and_get_env_vars()

        pod_deploy_commands = self._get_pod_deploy_commands(deploy_script)
        monitor_script = self._generate_pod_monitor_script()
        monitor_script_path = self._write_pod_monitor_script(monitor_script)
        graph_deploy_commands = self._get_graph_deploy_commands()
        # TODO: add port forwarding command
        # kubectl port-forward pod/$(kubectl get pods -n $NAMESPACE --no-headers | grep vllm-v1-agg-frontend | awk 'NR==1{print $1}') 8000:8000 -n $NAMESPACE
        # TODO: add curl commands to test the deployment
        # curl -X POST http://localhost:8000/v1/chat/completions

        all_commands = [
            *pod_deploy_commands,
            f"bash {monitor_script_path}",
            *graph_deploy_commands,
        ]
        cmd_parts = [*env_vars, f"( {' && '.join(all_commands)} ) > {stdout_path} 2> {stderr_path}"]

        final_cmd = " && ".join(cmd_parts)

        command_file = (self.test_run.output_path / "command.txt").absolute()
        with command_file.open("w") as f:
            f.write(final_cmd)

        return final_cmd

    def store_test_run(self) -> None:
        pass
