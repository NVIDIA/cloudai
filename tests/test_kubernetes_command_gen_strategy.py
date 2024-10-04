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

from pathlib import Path

import pytest
from cloudai.schema.test_template.nemo_launcher.kubernetes_command_gen_strategy import (
    NeMoLauncherKubernetesCommandGenStrategy,
)
from cloudai.systems.kubernetes import KubernetesSystem


@pytest.fixture
def kube_config_tempfile():
    kube_config_content = """
    apiVersion: v1
    kind: Config
    clusters:
    - cluster:
        server: https://127.0.0.1:6443
      name: local-cluster
    contexts:
    - context:
        cluster: local-cluster
        user: local-user
      name: local-context
    current-context: local-context
    users:
    - name: local-user
      user:
        token: fake-token
    """

    home_dir = Path.home()
    kube_config_dir = home_dir / ".kube"
    kube_config_path = kube_config_dir / "config"

    kube_config_dir.mkdir(parents=True, exist_ok=True)

    with kube_config_path.open("w") as config_file:
        config_file.write(kube_config_content)

    yield kube_config_path


@pytest.fixture
def k8s_system(kube_config_tempfile):
    system_data = {
        "name": "test-system",
        "install_path": "/fake/install/path",
        "output_path": "/fake/output/path",
        "kube_config_path": kube_config_tempfile,
        "default_namespace": "default",
        "default_image": "test-image",
        "scheduler": "kubernetes",
        "global_env_vars": {},
        "monitor_interval": 1,
    }

    k8s_system = KubernetesSystem(**system_data)
    k8s_system.model_post_init(None)

    validated_system = KubernetesSystem.model_validate(system_data)

    yield validated_system


class TestNeMoLauncherKubernetesCommandGenStrategy__GenExecCommand:
    @pytest.fixture
    def nemo_cmd_gen(self, k8s_system: KubernetesSystem) -> NeMoLauncherKubernetesCommandGenStrategy:
        cmd_args = {"test_arg": "test_value"}
        strategy = NeMoLauncherKubernetesCommandGenStrategy(k8s_system, cmd_args)
        strategy.system = k8s_system
        strategy.default_cmd_args = {
            "repository_url": "mock_repo_url",
            "repository_commit_hash": "mock_commit_hash",
            "docker_image_url": "mock_docker_image",
            "data_dir": "mock_data_dir",
            "training.values": "mock_training_values",
            "training.model.data.data_prefix": "\\mock_prefix",
        }
        return strategy

    def test_custom_gen_exec_command(self, nemo_cmd_gen: NeMoLauncherKubernetesCommandGenStrategy):
        extra_env_vars = {}
        cmd_args = {
            "launcher_scripts_path": "/fake/install/path/NeMoLauncherKubernetesInstallStrategy/launcher_scripts",
            "data_dir": "/nemo-workspace/pile",
            "cluster.value": "k8s_v2",
            "cluster.volumes.workspace.persistent_volume_claim.claim_name": "nemo-workspace",
            "cluster.volumes.workspace.mount_path": "/nemo-workspace",
            "stages": "[training]",
            "training.values": "gpt3/1b",
            "training.exp_manager.explicit_log_dir": "/nemo-workspace/gpt3/1b/training_gpt3/1b/results",
            "training.model.data.data_prefix": "[0.5,/nemo-workspace/pile/my-gpt3_00_text_document]",
            "training.trainer.num_nodes": "3",
            "training.trainer.devices": "8",
            "training.trainer.max_steps": "1000",
            "training.trainer.val_check_interval": "100",
            "training.model.global_batch_size": "48",
        }

        cmd = nemo_cmd_gen.gen_exec_command(
            cmd_args=cmd_args,
            extra_env_vars=extra_env_vars,
            extra_cmd_args="",
            output_path=Path("/fake/output/path"),
            num_nodes=3,
            nodes=["node1", "node2", "node3"],
        )

        expected_parts = [
            "python /fake/install/path/NeMo-Launcher/NeMo-Launcher/launcher_scripts/main.py",
            "data_dir=/nemo-workspace/pile",
            'training.model.data.data_prefix="[0.5,/nemo-workspace/pile/my-gpt3_00_text_document]"',
            "launcher_scripts_path=/fake/install/path/NeMo-Launcher/NeMo-Launcher/launcher_scripts",
            "cluster=k8s_v2",
            "cluster.volumes.workspace.persistent_volume_claim.claim_name=nemo-workspace",
            "cluster.volumes.workspace.mount_path=/nemo-workspace",
            'stages="[training]"',
            "training.exp_manager.explicit_log_dir=/nemo-workspace/gpt3/1b/training_gpt3/1b/results",
            "training.trainer.num_nodes=3",
            "training.trainer.devices=8",
            "training.trainer.max_steps=1000",
            "training.trainer.val_check_interval=100",
            "training.model.global_batch_size=48",
            "training=gpt3/1b",
        ]

        for part in expected_parts:
            assert part in cmd, f"Part '{part}' was not found in the generated command."
