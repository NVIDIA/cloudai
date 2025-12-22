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

import csv
from pathlib import Path

import pytest
import toml
import yaml

from cloudai.core import CommandGenStrategy, TestRun
from cloudai.models.scenario import TestRunDetails
from cloudai.models.workload import CmdArgs, TestDefinition
from cloudai.systems.kubernetes import KubernetesSystem
from cloudai.systems.runai import RunAISystem
from cloudai.systems.slurm import SlurmGroup, SlurmPartition, SlurmSystem
from cloudai.systems.standalone import StandaloneSystem
from cloudai.workloads.nccl_test.nccl import NCCLCmdArgs, NCCLTestDefinition


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield

    for f in {"env_vars.sh", "hostfile.txt", "start_server_wrapper.sh"}:
        (Path.cwd() / f).unlink(missing_ok=True)


@pytest.fixture
def slurm_system(tmp_path: Path) -> SlurmSystem:
    system = SlurmSystem(
        name="test_system",
        install_path=tmp_path / "install",
        output_path=tmp_path / "output",
        cache_docker_images_locally=True,
        default_partition="main",
        gpus_per_node=8,
        partitions=[
            SlurmPartition(
                name="main",
                groups=[
                    SlurmGroup(name="group1", nodes=["node-[033-048]"]),
                    SlurmGroup(name="group2", nodes=["node-[049-064]"]),
                ],
            ),
            SlurmPartition(
                name="backup",
                groups=[
                    SlurmGroup(name="group1", nodes=["node0[1-4]"]),
                    SlurmGroup(name="group2", nodes=["node0[5-8]"]),
                ],
            ),
        ],
    )
    system.scheduler = "slurm"
    system.monitor_interval = 0
    return system


@pytest.fixture
def k8s_system(tmp_path: Path) -> KubernetesSystem:
    kube_config = tmp_path / "kubeconfig"
    config = {
        "apiVersion": "v1",
        "kind": "Config",
        "current-context": "test-context",
        "clusters": [{"name": "test-cluster", "cluster": {"server": "https://test-server:6443"}}],
        "contexts": [{"name": "test-context", "context": {"cluster": "test-cluster", "user": "test-user"}}],
        "users": [{"name": "test-user", "user": {"token": "test-token"}}],
    }
    kube_config.write_text(yaml.dump(config))

    system = KubernetesSystem(
        name="test_kubernetes_system",
        install_path=tmp_path / "install",
        output_path=tmp_path / "output",
        scheduler="kubernetes",
        global_env_vars={},
        monitor_interval=1,
        gpus_per_node=8,
        default_namespace="test-namespace",
        kube_config_path=kube_config,
    )
    return system


@pytest.fixture
def runai_system(tmp_path: Path) -> RunAISystem:
    system = RunAISystem(
        name="test_runai_system",
        install_path=tmp_path / "install",
        output_path=tmp_path / "output",
        base_url="http://runai.example.com",
        app_id="test_app_id",
        app_secret="test_app_secret",
        project_id="test_project_id",
        cluster_id="test_cluster_id",
        scheduler="runai",
        global_env_vars={},
        monitor_interval=60,
        user_email="test_user@example.com",
    )
    return system


@pytest.fixture
def standalone_system(tmp_path: Path) -> StandaloneSystem:
    return StandaloneSystem(
        name="standalone",
        scheduler="standalone",
        install_path=tmp_path / "install",
        output_path=tmp_path / "output",
    )


@pytest.fixture
def base_tr(slurm_system: SlurmSystem) -> TestRun:
    return TestRun(
        name="tr-name",
        test=TestDefinition(name="n", description="d", test_template_name="tt", cmd_args=CmdArgs()),
        num_nodes=1,
        nodes=[],
        output_path=slurm_system.output_path / "tr-name",
    )


def create_test_directories(slurm_system: SlurmSystem, test_run: TestRun) -> None:
    test_dir = slurm_system.output_path / test_run.name
    for iteration in range(test_run.iterations):
        folder = test_dir / str(iteration)
        folder.mkdir(exist_ok=True, parents=True)
        if test_run.is_dse_job:
            with open(folder / "trajectory.csv", "w") as _f_csv:
                csw_writer = csv.writer(_f_csv)
                csw_writer.writerow(["step", "action", "reward", "observation"])

                for step in range(test_run.test.agent_steps):
                    step_folder = folder / str(step)
                    step_folder.mkdir(exist_ok=True, parents=True)
                    trd = TestRunDetails.from_test_run(test_run, "", "")
                    csw_writer.writerow([step, {}, step * 2.1, [step]])
                    with open(step_folder / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME, "w") as _f_trd:
                        toml.dump(trd.model_dump(), _f_trd)


@pytest.fixture
def benchmark_tr(slurm_system: SlurmSystem) -> TestRun:
    test_definition = NCCLTestDefinition(
        name="nccl",
        description="NCCL test",
        test_template_name="NcclTest",
        cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl"),
    )
    tr = TestRun(name="benchmark", test=test_definition, num_nodes=1, nodes=["node1"], iterations=3)
    create_test_directories(slurm_system, tr)
    return tr


@pytest.fixture
def dse_tr(slurm_system: SlurmSystem) -> TestRun:
    test_definition = NCCLTestDefinition(
        name="nccl",
        description="NCCL test",
        test_template_name="NcclTest",
        cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl"),
        extra_env_vars={"VAR1": ["value1", "value2"]},
        agent_steps=12,
    )

    tr = TestRun(name="dse", test=test_definition, num_nodes=1, nodes=["node1"], iterations=12)
    create_test_directories(slurm_system, tr)
    return tr
